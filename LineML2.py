
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.transforms import RandomLinkSplit, LineGraph
from torch_geometric.loader import LinkNeighborLoader
from pytorch_metric_learning import losses
import pickle
from torch_geometric.utils import negative_sampling
from torch_geometric.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score

from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes, aggr='mean', dropout_rate=0.7):
        super(GraphSAGE, self).__init__()
        
        # Layer 1
        self.conv1 = SAGEConv(num_features, num_features*2, aggr=aggr)  
        # Layer 2
        self.conv2 = SAGEConv(num_features*2, num_features, aggr=aggr)  
        # Layer 3
        self.conv3 = SAGEConv(num_features, num_classes, aggr=aggr)  

        # Dropout layer
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
       
        
        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)  # Apply dropout

        # Second layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)  # Apply dropout
        
        # Third layer
        x = self.conv3(x, edge_index)
        
        return F.log_softmax(x, dim=1)
class Classifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()

        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3 = torch.nn.Linear(hidden_dim//2, hidden_dim//4)
        self.fc4 = torch.nn.Linear(hidden_dim//4, output_dim)
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
      
        x = self.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.relu(self.fc2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.relu(self.fc3(x))
        x = F.dropout(x, p=0.2, training=self.training)
        # Apply the final linear layer
        x = self.fc4(x)
        # Apply softmax activation to get probabilities
        x = self.sigm(x)
        return x

def process_batch(batch):
    print(batch)

   
    t=batch.clone()
    
    transform = LineGraph()
    line_graph = transform(t)
    print(line_graph)
   
    
    # Step 2: Extract features for the line graph
    # The feature of each edge (now a node in the line graph) is the concatenation
    # of the features of the two nodes connected by the edge in the original graph
    
    edge_features = torch.cat([
        torch.cat([batch.x[batch.edge_index[0]], batch.x[batch.edge_index[1]]], dim=1)
    ], dim=0)
    
    # Step 4: Map the original graph's edges to line graph nodes
    # The original graph's edges are indexed in batch.edge_index. Each edge (u, v)
    # becomes a node in the line graph. So we need to assign labels based on edge indices.
    num_edges = batch.edge_index.size(1)
    node_label_line_graph = torch.full((num_edges,), -1, dtype=torch.long, device=batch.edge_index.device)
    print(f"num_edges: {num_edges}, node_label_line_graph: {node_label_line_graph.shape} edge_label_index: {batch.edge_label_index.shape}, edge_label: {batch.edge_label_index.shape}")
    # Use edge_label_index to assign edge labels to the corresponding line graph nodes
    # node_label_line_graph[batch.edge_label_index[0]] = batch.edge_label.long()
    # Number of edges
    num_edges = batch.edge_index.size(1)

    # Initialize labels with -1
    edge_labels = torch.full((num_edges,), -1, dtype=torch.int)

    # Assign labels to edges
    for i in range(batch.edge_label_index.size(1)):
        source = batch.edge_label_index[0, i].item()
        target = batch.edge_label_index[1, i].item()

        # Find the index of the edge in edge_index
        edge_found = ((batch.edge_index[0] == source) & (batch.edge_index[1] == target)).nonzero(as_tuple=True)
        
        if edge_found[0].numel() > 0:
            edge_index_pos = edge_found[0].item()  # Get the index of the edge
            edge_labels[edge_index_pos] = batch.edge_label[i]
    print(edge_labels)
    exit(0)
    # Construct the line graph batch with nodes corresponding to the edges of the original graph
    if hasattr(batch, 'n_id'):
        del batch.n_id  # Remove edge_label attribute if it exists
    if hasattr(batch, 'e_id'):
        del batch.e_id  # Remove edge_label attribute if it exists
    if hasattr(batch, 'input_id'):
        del batch.input_id  # Remove edge_label attribute if it exists
    batch.x_line=edge_features
    batch.edge_index_line=line_graph.edge_index
    batch.nods_label=node_label_line_graph

        # Step 4: Initialize a new edge label array with -1 for all edges
    num_edges = batch.edge_index.size(1)  # This is 64 in your case
    full_edge_label = torch.full((num_edges,), -1, dtype=torch.long, device=batch.edge_index.device)

    # Step 5: Map the existing edge labels to their respective positions using edge_label_index
    full_edge_label[batch.edge_label_index[1]] = batch.edge_label.long()
    batch.edge_label=full_edge_label
    if hasattr(batch, 'edge_label_index'):
        del batch.edge_label_index  # Remove edge_label attribute if it exists
    return batch # Returning both original graph and line graph data

def count_edge_labels(data):
    # Get the edge labels from the dataset (positive edges: 1, negative edges: 0)
    edge_label = data.edge_label
    
    # Count the number of 1's and 0's
    num_ones = torch.sum(edge_label == 1).item()
    num_zeros = torch.sum(edge_label == 0).item()
    
    return num_zeros, num_ones

def Get_data_loaders(dataset, num_val=0.05, num_test=0.1, disjoint_train_ratio=0.8, neg_sampling_ratio=1, batch_size=256):

    transformer = RandomLinkSplit(
        num_val=num_val,
        num_test=num_test,
        disjoint_train_ratio=disjoint_train_ratio,
        neg_sampling_ratio=neg_sampling_ratio,
        add_negative_train_samples=True,
        split_labels=False)

    train_data, val_data, test_data = transformer(dataset)
    print(train_data)

    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[-1],
        neg_sampling_ratio=0,
        edge_label_index=train_data.edge_label_index,
        edge_label=train_data.edge_label,
        batch_size=batch_size,
        shuffle=True,
    )



    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=[-1],
        neg_sampling_ratio=0,
        edge_label_index=val_data.edge_label_index,
        edge_label=val_data.edge_label,
        batch_size=batch_size,
        shuffle=True,
    )


    test_loader = LinkNeighborLoader(
        data=test_data,
        num_neighbors=[-1],
        neg_sampling_ratio=0,
        edge_label_index=test_data.edge_label_index,
        edge_label=test_data.edge_label,
        batch_size=batch_size,
        shuffle=True,
    )

     
    return train_loader, val_loader, test_loader

def generate_triplet_samples(embeddings, y):
    """
    Generate triplet samples for triplet loss from embeddings and edge labels.
    
    Parameters:
        embeddings (torch.Tensor): The embeddings produced by the GNN gnn.
        edge_labels (torch.Tensor): The labels indicating positive (1) and negative (0) edges.

    Returns:
        tuple: A tuple containing anchor, positive, and negative samples.
    """
    
    positive_indices = np.where(y.cpu().numpy() == 1.0)[0]  # Indices of positive samples
    negative_indices = np.where(y.cpu().numpy() == 0.0)[0]  # Indices of negative samples
    # Randomly select triplet samples
    num_triplets = min(len(positive_indices), len(negative_indices))
    
    if num_triplets == 0:
        raise ValueError("Not enough positive or negative samples to generate triplets.")

    # Shuffle positive and negative indices
    np.random.shuffle(positive_indices)
    np.random.shuffle(negative_indices)

    # Randomly select triplet samples
    anchor_samples = positive_indices[:num_triplets]  # Select first 'num_triplets' from shuffled positives
    np.random.shuffle(positive_indices)
    positive_samples = positive_indices[:num_triplets] 
    negative_samples = negative_indices[:num_triplets]        # Select first 'num_triplets' from shuffled negatives

    anchor = embeddings[anchor_samples]
    positive = embeddings[positive_samples]  # Same as anchor
    negative = embeddings[negative_samples]

    return anchor, positive, negative


# def train_dual_gnn_model(train_loader,  epochs=100,gnn_aggr='mean',alpha=1.0,lr_lc=0.0000001,gnn_lr=0.001, device='cpu'):

#     model_gnn1 = GraphSAGE(64, 32, aggr=gnn_aggr, dropout_rate=0.2)
#     optimizer=torch.optim.Adam(model_gnn1.parameters(), lr=gnn_lr)
#     # triplet_loss_fn = torch.nn.TripletMarginLoss(margin=alpha)  # Adjust margin as needed
#     gnn1_loss = F.cross_entropy
#     model_gnn1.to(device)
#     model_gnn1.train()

#     model_gnn2 = GraphSAGE(128, 64, aggr=gnn_aggr, dropout_rate=0.2)
#     optimizer=torch.optim.Adam(model_gnn2.parameters(), lr=gnn_lr)
#     # triplet_loss_fn = torch.nn.TripletMarginLoss(margin=alpha)  # Adjust margin as needed
#     gnn2_loss = F.cross_entropy
#     model_gnn2.to(device)
#     model_gnn2.train()


#     lc = Classifier(128, 32, 1)
#     lc = lc.to(device)
#     lc_optim = torch.optim.Adam(lc.parameters(), lr=lr_lc)
#     lc_loss = nn.BCELoss()
#     lc.to(device)
#     lc.train()
#     model_gnn1.train()
#     model_gnn2.train()
#     lc.train()

#     for epoch in range(epochs):
#         for data in train_loader:
#             data = process_batch(data)
            
#             optimizer.zero_grad()

#             # Forward pass through GNN 1
#             node_embeddings_gnn1 = model_gnn1(data.x, data.edge_index)  # GNN 1 output
#             link_embeddings_gnn1 = torch.cat([node_embeddings_gnn1[data.edge_index[0]],
#                                             node_embeddings_gnn1[data.edge_index[1]]], dim=-1)  # Concatenate node embeddings to form link embeddings

#             # Filter out unlabeled edges based on edge_label == -1
#             mask_labeled_edges = data.edge_label != -1
            
#             link_embeddings_gnn1_filtered = link_embeddings_gnn1[mask_labeled_edges]
#             edge_label_filtered = data.edge_label[mask_labeled_edges]
#             loss1= gnn1_loss(link_embeddings_gnn1_filtered, edge_label_filtered, reduction='sum')
#             # Forward pass through GNN 2 (on line graph)
#             node_embeddings_gnn2 = model_gnn2(data.x_line, data.edge_index_line)  # GNN 2 output
#             node_embeddings_gnn2_filtered = node_embeddings_gnn2[mask_labeled_edges]
#             loss2= gnn2_loss(node_embeddings_gnn2_filtered, edge_label_filtered, reduction='sum')
#             loss1.backward(retain_graph=True)
#             loss2.backward(retain_graph=True)
#             # Concatenate link embeddings from GNN 1 and node embeddings from GNN 2
#             combined_embeddings = torch.cat([link_embeddings_gnn1_filtered, node_embeddings_gnn2_filtered], dim=-1)

#             # Forward pass through MLP
#             predictions = lc(combined_embeddings)

#             # Calculate the loss
#             loss = lc_loss(predictions.squeeze(), edge_label_filtered.float())

#             # Backward pass and optimization step
#             loss.backward()
#             optimizer.step()

#             # Logging the loss for each epoch
#             print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()} Loss gnn1 : {loss1.item()} Loss gnn2 : {loss2.item()}')


def train_dual_gnn_model(train_loader, val_loader, epochs=500, aggr='mean',gnn_aggr='max', alpha=1.0, lr_lc=0.00001, gnn_lr=0.001, device='cpu'):
    # Initialize GNN 1 and optimizer
    gnn1 = GraphSAGE(64, 32, aggr=gnn_aggr, dropout_rate=0.2).to(device)
    gnn1_optimizer = torch.optim.Adam(gnn1.parameters(), lr=gnn_lr)
    triplet_loss_fn1 = torch.nn.TripletMarginLoss(margin=alpha)  

    # Initialize GNN 2 and optimizer
    gnn2 = GraphSAGE(128, 64, aggr=gnn_aggr, dropout_rate=0.2).to(device)
    gnn2_optimizer = torch.optim.Adam(gnn2.parameters(), lr=gnn_lr)
    triplet_loss_fn2 = torch.nn.TripletMarginLoss(margin=alpha)  

    # Initialize Link Classifier (MLP) and optimizer
    if aggr == 'cat':
        lc_input_dim = 64 + 64
    else:   
        lc_input_dim = 64
    link_classifier = Classifier(lc_input_dim, 32, 1).to(device)
    lc_optimizer = torch.optim.Adam(link_classifier.parameters(), lr=lr_lc)
    lc_loss_fn = nn.BCELoss()

    # Training Loop
    for epoch in range(epochs):
        gnn1.train()
        gnn2.train()
        link_classifier.train()

        total_loss = 0
        total_loss_gnn1 = 0
        total_loss_gnn2 = 0

        all_train_predictions = []
        all_train_labels = []

        # Training phase
        for batch in train_loader:
            batch = process_batch(batch)

            # Zero gradients
            gnn1_optimizer.zero_grad()
            gnn2_optimizer.zero_grad()
            lc_optimizer.zero_grad()

            # Forward pass through GNN 1 (original graph)
            node_embeddings_gnn1 = gnn1(batch.x, batch.edge_index)
            link_embeddings_gnn1 = torch.cat([node_embeddings_gnn1[batch.edge_index[0]], 
                                              node_embeddings_gnn1[batch.edge_index[1]]], dim=-1)

            # Filter labeled edges for GNN 1
            mask_labeled_edges = batch.edge_label != -1
            link_embeddings_gnn1_filtered = link_embeddings_gnn1[mask_labeled_edges]
            edge_labels_filtered = batch.edge_label[mask_labeled_edges]

            # Forward pass through GNN 2 (line graph)
            node_embeddings_gnn2 = gnn2(batch.x_line, batch.edge_index_line)
            node_embeddings_gnn2_filtered = node_embeddings_gnn2[mask_labeled_edges]

            anchor, positive, negative = generate_triplet_samples(link_embeddings_gnn1_filtered, edge_labels_filtered)
            
            # Calculate triplet loss
            loss_gnn1 = triplet_loss_fn1(anchor, positive, negative)
            # Loss calculation for GNNs

            anchor, positive, negative = generate_triplet_samples(node_embeddings_gnn2_filtered, edge_labels_filtered)
            
            # Calculate triplet loss
            loss_gnn2 = triplet_loss_fn1(anchor, positive, negative)

            # Backpropagate GNN losses
            loss_gnn1.backward(retain_graph=True)
            loss_gnn2.backward(retain_graph=True)

            # Concatenate embeddings from GNN 1 and GNN 2
            if aggr == 'cat':
                combined_embeddings = torch.cat([link_embeddings_gnn1_filtered, node_embeddings_gnn2_filtered], dim=-1)
            if aggr == 'mean':
                combined_embeddings = torch.mean(torch.stack([link_embeddings_gnn1_filtered, node_embeddings_gnn2_filtered]), dim=0)
  
            # Forward pass through MLP (link classifier)
            predictions = link_classifier(combined_embeddings).squeeze()
           
          
            # Loss calculation for link classifier
            loss_lc = lc_loss_fn(predictions, edge_labels_filtered.float())

            # Backpropagate link classifier loss
            loss_lc.backward()
            
            # Optimizer steps
            lc_optimizer.step()
            gnn1_optimizer.step()
            gnn2_optimizer.step()
            

            # Logging loss values
            total_loss += loss_lc.item()
            total_loss_gnn1 += loss_gnn1.item()
            total_loss_gnn2 += loss_gnn2.item()

            # Store predictions and labels for ROC AUC calculation
            all_train_predictions.append(predictions.cpu().detach().numpy())
            all_train_labels.append(edge_labels_filtered.cpu().detach().numpy())

        # Compute training ROC AUC
        all_train_predictions = np.concatenate(all_train_predictions)
        all_train_labels = np.concatenate(all_train_labels)
        train_roc_auc = roc_auc_score(all_train_labels, all_train_predictions)
        total_loss = total_loss / len(train_loader)
        total_loss_gnn1 = total_loss_gnn1 / len(train_loader)
        total_loss_gnn2 = total_loss_gnn2 / len(train_loader)
        

        # Validation phase
        gnn1.eval()
        gnn2.eval()
        link_classifier.eval()

        all_val_predictions = []
        all_val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                batch = process_batch(batch)
                mask_labeled_edges = batch.edge_label != -1
                edge_labels_filtered = batch.edge_label[mask_labeled_edges]

                # Forward pass through GNNs
                node_embeddings_gnn1 = gnn1(batch.x, batch.edge_index)
                link_embeddings_gnn1 = torch.cat([node_embeddings_gnn1[batch.edge_index[0]], 
                                                  node_embeddings_gnn1[batch.edge_index[1]]], dim=-1)

                node_embeddings_gnn2 = gnn2(batch.x_line, batch.edge_index_line)
                node_embeddings_gnn2_filtered = node_embeddings_gnn2[mask_labeled_edges]

                # Concatenate link embeddings from GNN 1 and node embeddings from GNN 2
               
                if aggr == 'cat':
                    combined_embeddings = torch.cat([link_embeddings_gnn1[mask_labeled_edges], node_embeddings_gnn2_filtered], dim=-1)
                if aggr == 'mean':
                    combined_embeddings = torch.mean(torch.stack([link_embeddings_gnn1[mask_labeled_edges],  node_embeddings_gnn2_filtered]), dim=0)                
                # Forward pass through MLP
                val_predictions = link_classifier(combined_embeddings).squeeze()

                # Store predictions and labels for ROC AUC calculation
                all_val_predictions.append(val_predictions.cpu().numpy())
                all_val_labels.append(edge_labels_filtered.cpu().numpy())

        # Compute validation ROC AUC
        all_val_predictions = np.concatenate(all_val_predictions)
        all_val_labels = np.concatenate(all_val_labels)
        val_roc_auc = roc_auc_score(all_val_labels, all_val_predictions)
        print(f'Epoch {epoch+1}/{epochs}, Loss GNN1: {total_loss_gnn1:.4f}, Loss GNN2: {total_loss_gnn2:.4f}, Loss LC: {total_loss:.4f}, Train ROC AUC: {train_roc_auc:.4f} Validation ROC AUC: {val_roc_auc:.4f}')


with open(f"data/datasets/processed/BP.pkl", "rb") as file:
        data = pickle.load(file)

train_loader, val_loader, test_loader = Get_data_loaders(data)
train_dual_gnn_model(train_loader,val_loader)