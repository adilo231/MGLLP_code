import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import RandomNodeSplit, LineGraph
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import pandas as pd
import os

class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes, aggr='mean', dropout_rate=0.7):
        super(GraphSAGE, self).__init__()
        
        # Layer 1
        self.conv1 = SAGEConv(num_features, 256, aggr=aggr)  
        # Layer 2
        self.conv2 = SAGEConv(256, 128, aggr=aggr)  
        # Layer 3
        self.conv3 = SAGEConv(128, num_classes, aggr=aggr)  

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
        self.fc3 = torch.nn.Linear(hidden_dim//2, output_dim)
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
      
        x = self.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.relu(self.fc2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        # Apply the final linear layer
        x = self.fc3(x)
        # Apply softmax activation to get probabilities
        x = self.sigm(x)
        return x


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


def get_data_loaders(dataset, num_val=0.1, num_test=0.1, disjoint_train_ratio=0.8, neg_sampling_ratio=1.0, batch_size=32):
    """
    Creates train, validation, and test data loaders for the line graph with positive and negative edge labels.
    
    Parameters:
        dataset (torch_geometric.data.Data): The input graph dataset.
        num_val (float): Proportion of the data to use for validation.
        num_test (float): Proportion of the data to use for testing.
        disjoint_train_ratio (float): Proportion of the training data without test/validation edges.
        neg_sampling_ratio (float): Ratio of negative to positive samples.
        batch_size (int): Batch size for the data loaders.

    Returns:
        train_loader, val_loader, test_loader (DataLoader): Data loaders for train, validation, and test.
    """
    original_node_features = dataset.x 
    
    # Step 1: Negative Sampling before transforming to line graph
    neg_edge_index = negative_sampling(
        edge_index=dataset.edge_index,  # Edge index from the original graph
        num_nodes=dataset.num_nodes,
        num_neg_samples=int(dataset.edge_index.size(1) * neg_sampling_ratio)  # Create negative edges with same ratio
    )
    
    # Combine positive and negative edges into one edge index
    combined_edge_index = torch.cat([dataset.edge_index, neg_edge_index], dim=1)
    
    # Create edge labels: Positive edges labeled as 1, negative edges as 0
    pos_labels = torch.ones(dataset.edge_index.size(1))  # Positive edge labels
    neg_labels = torch.zeros(neg_edge_index.size(1))  # Negative edge labels
    combined_edge_labels = torch.cat([pos_labels, neg_labels], dim=0)
    
    # Update the dataset with new edges and labels
    dataset.edge_index = combined_edge_index
    dataset.edge_label = combined_edge_labels
   
    # Step 2: Line Graph Transformation
    transform = LineGraph()
    line_graph = transform(dataset)
    max_value = torch.max(line_graph.edge_index)
   
    # Step 3: Split the edges into train, validation, and test sets
    
    
    line_graph.y = combined_edge_labels
    if hasattr(line_graph, 'edge_label'):
        del line_graph.edge_label  # Remove edge_label attribute if it exists


    
    line_graph_features = []  # List to hold the features for nodes in the line graph
  
    # Step 1: Iterate over the edges in the original graph
    for edge in combined_edge_index.t().tolist():  # Convert edge_index to list of edges
       
        node1, node2 = edge  # Get the two nodes connected by the edge
     
        # Step 2: Concatenate features of the two nodes
        concatenated_features = torch.cat((original_node_features[node1], original_node_features[node2]))
        line_graph_features.append(concatenated_features)  # Append the concatenated features

    # Step 3: Convert to a tensor and assign to line_graph
    line_graph.x = torch.stack(line_graph_features) 
    
    
    # Step 5: Split the nodes (not edges) into train, validation, and test sets
    node_split = RandomNodeSplit(num_val=num_val, num_test=num_test)
    line_graph = node_split(line_graph)  # Adds train_mask, val_mask, test_mask
    
    
    # Step 6: Create NeighborLoader for node-based sampling
    train_loader = NeighborLoader(
        data=line_graph,
        num_neighbors=[-1],  # Sample all neighbors for the train set
        input_nodes=line_graph.train_mask,  # Sample from nodes in the training mask
        batch_size=batch_size,
        shuffle=True,
    )
    # Now process each batch to subset the 'x' features
    for batch in train_loader:
        # batch.n_id gives the node IDs in the batch
        batch_x = line_graph.x[batch.n_id] 
    val_loader = NeighborLoader(
        data=line_graph,
        num_neighbors=[-1],  # Sample all neighbors for the validation set
        input_nodes=line_graph.val_mask,  # Sample from nodes in the validation mask
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = NeighborLoader(
        data=line_graph,
        num_neighbors=[-1],  # Sample all neighbors for the test set
        input_nodes=line_graph.test_mask,  # Sample from nodes in the test mask
        batch_size=batch_size,
        shuffle=False,
    )
    
    return train_loader, val_loader, test_loader




def train_with_triplet_loss( train_loader,val_loader,test_loader,gnn_num_input=128,gnn_num_output=64, gnn_aggr='mean',alpha=1.0,lr_lc=0.000001,gnn_lr=0.001, device='cpu',epochs=200,patience=50,dataset="EML"):
    
    gnn = GraphSAGE(gnn_num_input, gnn_num_output, aggr=gnn_aggr, dropout_rate=0.2)
    optimizer=torch.optim.Adam(gnn.parameters(), lr=gnn_lr)
    triplet_loss_fn = torch.nn.TripletMarginLoss(margin=alpha)  # Adjust margin as needed
    # gnn_loss = F.cross_entropy
    gnn.to(device)
    gnn.train()


    lc = Classifier(gnn_num_output, 32, 1)
    lc = lc.to(device)
    lc_optim = torch.optim.Adam(lc.parameters(), lr=lr_lc)
    lc_loss = nn.BCELoss()
    lc.to(device)
    lc.train()
    
    train_losses_gnn = []
    train_losses_lc = []
    roc_train = []
    Ap_train = []
    roc_val = []
    Ap_val = []

    best_val_roc = 0
    early_stop_counter = 0
    
    for epoch in range(epochs):  # Define 'epochs' as needed
       
        total_loss_gnn = 0.0
        total_loss_lc = 0.0
        train_preds,train_y= [],[]
        for data in train_loader:
            
            optimizer.zero_grad()
            data = data.to(device)

          
            
            # Get the node embeddings for the batch
            embeddings = gnn(data.x, data.edge_index)  # `data.n_id` refers to the nodes in the batch
            
            # Filter embeddings and labels for nodes in the training set
            train_embeddings = embeddings[data.train_mask]
            train_labels = data.y[data.train_mask]  # Assuming that `y` contains the labels of the nodes
            # train_embeddings = embeddings
            # train_labels = data.y  # Assuming that `y` contains the labels of the nodes
             # Generate triplet samples (anchor, positive, negative)
            anchor, positive, negative = generate_triplet_samples(train_embeddings, train_labels)
            
            # Calculate triplet loss
            loss = triplet_loss_fn(anchor, positive, negative)
            
            loss.backward(retain_graph=True)
            
            # loss = gnn_loss(train_embeddings ,train_labels.to(torch.long))
            # loss.backward(retain_graph=True)
            # if epoch == 50:
            #     print("First five embeddings with their labels:")
            #     for i in range(5):
            #         print("Embedding:", train_embeddings[i][:5].detach().cpu().numpy())
            #         print("Label:", train_labels[i].detach().cpu().numpy())
            #     print()
            
            pred = lc(train_embeddings)
          
            # Calculate classification loss
            loss2 = lc_loss(pred.squeeze(), train_labels)
            loss2.backward()

            
            lc_optim.step()
            optimizer.step()
           
            
            total_loss_gnn += loss.item()
            total_loss_lc += loss2.item()
            train_preds.append(pred)
            train_y.append(train_labels)

        # Concatenate embeddings and labels
        train_preds = torch.cat(train_preds, dim=0).cpu().detach().numpy()
        train_y = torch.cat(train_y, dim=0).cpu().numpy()
        
        # Compute ROC AUC and Average Precision
       
        train_roc_auc = roc_auc_score(train_y, train_preds)  # Adjust based on your embedding structure
        train_ap = average_precision_score(train_y, train_preds)  # Adjust based on your embedding structure
        roc_train.append(train_roc_auc)
        Ap_train.append(train_ap)
        
        avg_train_loss_gnn = total_loss_gnn / len(train_loader)
        avg_train_loss_lc = total_loss_lc / len(train_loader)
        train_losses_gnn.append(avg_train_loss_gnn)
        train_losses_lc.append(avg_train_loss_lc)
        
        # Validation
        gnn.eval()
        lc.eval()
        val_preds, val_labels = [], []
        # try:
        with torch.no_grad():
            for val_data in val_loader:
                
                val_data = val_data.to(device)
                val_emb = gnn(val_data.x, val_data.edge_index)
                
                # Filter embeddings and labels for nodes in the validation set
                val_emb = val_emb[val_data.val_mask]
               
                pred = lc(val_emb)
                
                val_y = val_data.y
                val_y = val_data.y[val_data.val_mask]  # Assuming that `y` contains the labels of the nodes
                # Append the selected embeddings and labels
                val_preds.append(pred)
                val_labels.append(val_y)

        # Concatenate embeddings and labels
        val_preds = torch.cat(val_preds, dim=0).cpu().numpy()
        val_labels = torch.cat(val_labels, dim=0).cpu().numpy()
        
        # Compute ROC AUC and Average Precision
       
        val_roc_auc = roc_auc_score(val_labels, val_preds)  # Adjust based on your embedding structure
        val_ap = average_precision_score(val_labels, val_preds)  # Adjust based on your embedding structure
        roc_val.append(val_roc_auc)
        Ap_val.append(val_roc_auc)
        # print(f'Epoch: {epoch + 1}, Train gnn Loss: {avg_train_loss_gnn:.4f}, Train lc Loss:: {avg_train_loss_lc:.4f}, Train ROC : {train_roc_auc:.4f}, Val ROC : {val_roc_auc:.4f}, Val AP: {val_ap:.4f}')

               
        # Early stopping
        
        if val_roc_auc > best_val_roc:
            best_val_roc = val_roc_auc
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if (val_roc_auc>.60 and early_stop_counter >= patience) or (val_roc_auc<=.60 and early_stop_counter >= 5*patience):
            # print(f"Early stopping triggered after {epoch + 1} epochs.")
            break
        # except Exception as e:
        #     print(f'Error in epoch {epoch + 1}: {str(e)}')
        # Plot the losses
    roc_test, ap_test = evaluate_on_test_set(gnn, lc, test_loader, device)
    plots_dir = f"plots/{dataset}"
    os.makedirs(plots_dir, exist_ok=True)

    # Prepare the hyperparameters text to be displayed and included in file names
    hyperparameters_text = (
        f"alpha={alpha}, gnn_lr={gnn_lr}, lc_lr={lr_lc}\n"
        f"gnn_num_output={gnn_num_output}, epochs={epochs}, patience={patience}"
    )
    file_name_suffix = f"alpha_{alpha}_gnn_lr_{gnn_lr}_lc_lr_{lr_lc}"

    # Save training loss plot with hyperparameters
    plt.figure()
    plt.plot(train_losses_gnn, label='gnn_train_loss')
    plt.plot(train_losses_lc, label='lc_train_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.figtext(0.5, 0.01, hyperparameters_text, wrap=True, horizontalalignment='center', fontsize=8)
    plt.legend()
    loss_plot_path = os.path.join(plots_dir, f"training_losses_{file_name_suffix}.pdf")
    plt.savefig(loss_plot_path, format='pdf')
    plt.close()

    # Save ROC plot with hyperparameters
    plt.figure()
    plt.plot(roc_train, label='roc train')
    plt.plot(roc_val, label='roc val')
    plt.xlabel('Epoch')
    plt.ylabel('ROC')
    plt.title('Training and Validation ROC')
    plt.figtext(0.5, 0.01, hyperparameters_text, wrap=True, horizontalalignment='center', fontsize=8)
    plt.legend()
    roc_plot_path = os.path.join(plots_dir, f"roc_plot_{file_name_suffix}.pdf")
    plt.savefig(roc_plot_path, format='pdf')
    plt.close()
    return roc_test, ap_test,epoch
            

def evaluate_on_test_set(gnn, lc, test_loader, device):
    gnn.eval()  # Set the GNN model to evaluation mode
    lc.eval()   # Set the classifier to evaluation mode
    
    test_preds, test_labels = [], []
    

    with torch.no_grad():  # No need to calculate gradients during evaluation
        for data in test_loader:
            data = data.to(device)
            
            # GNN forward pass to get node embeddings
            test_emb = gnn(data.x, data.edge_index)
            
            # Filter embeddings and labels for nodes in the test set
            test_emb = test_emb[data.test_mask]
            test_y = data.y[data.test_mask]
            

            # Classifier forward pass
            test_pred = lc(test_emb)

            # Collect predictions and labels for metric calculation
            test_preds.append(test_pred)
            test_labels.append(test_y)

    # Concatenate all predictions and labels across the batches
    test_preds = torch.cat(test_preds, dim=0).cpu().numpy()
    test_labels = torch.cat(test_labels, dim=0).cpu().numpy()

    # Compute ROC AUC and Average Precision scores
    test_roc_auc = roc_auc_score(test_labels, test_preds)
    test_ap = average_precision_score(test_labels, test_preds)
    

    # print(f"Test ROC AUC: {test_roc_auc:.4f}, Test AP: {test_ap:.4f}")
    
    return test_roc_auc, test_ap
    

def main():

    parser = argparse.ArgumentParser(description="Train a GNN model with triplet loss")
    
    # Add command-line arguments
    parser.add_argument('--dataset', type=str, default='NSC', help="Dataset name")
    parser.add_argument('--gnn_num_input', type=int, default=32, help="Number of input features for the GNN")
    parser.add_argument('--gnn_num_output', type=int, default=32, help="Number of output features for the GNN")
    parser.add_argument('--gnn_aggr', type=str, default='mean', help="Aggregation method for the GNN")
    parser.add_argument('--alpha', type=float, default=1.0, help="Alpha value for the triplet loss")
    parser.add_argument('--lr_lc', type=float, default=0.000001, help="Learning rate for the link classifier")
    parser.add_argument('--gnn_lr', type=float, default=0.001, help="Learning rate for the GNN")
    parser.add_argument('--epochs', type=int, default=1000, help="Number of training epochs")
    parser.add_argument('--patience', type=int, default=20, help="Patience for early stopping")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for data loaders")
    # Check if CUDA is available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} device")
    # Parse arguments
    args = parser.parse_args()
    print(f"Starting expriments for {args.dataset}, aggr={args.gnn_aggr}, alpha={args.alpha}, lr_lc={args.lr_lc}, gnn_lr={args.gnn_lr}, epochs={args.epochs}, patience={args.patience}, batch_size={args.batch_size}")
    # Load dataset
    with open(f"data/datasets/processed/{args.dataset}.pkl", "rb") as file:
        data = pickle.load(file)

    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(data, batch_size=args.batch_size)

    # Train the model
    ROC_test,AP_test,epoch_needed=train_with_triplet_loss(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        gnn_num_input=data.num_features,
        gnn_num_output=args.gnn_num_output,
        gnn_aggr=args.gnn_aggr,
        alpha=args.alpha,
        lr_lc=args.lr_lc,
        gnn_lr=args.gnn_lr,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        dataset=args.dataset
    )
    file_path = f"results.csv"
    file_exists = os.path.isfile(file_path)
    
    # Prepare the dictionary of results and hyperparameters
    results = {
        'dataset': [args.dataset],
        'alpha': [args.alpha],
        'lr_lc': [args.lr_lc],
        'gnn_lr': [args.gnn_lr],
        'batch_size': [args.batch_size],
        'gnn_aggr': [args.gnn_aggr],
        'ROC_test': [ROC_test],
        'AP_test': [AP_test],
        'epoch_needed': [epoch_needed],
        'epochs': [args.epochs],
        'patience': [args.patience],
        'gnn_num_input': [args.gnn_num_input],
        'gnn_num_output': [args.gnn_num_output],
    }
    
    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(results)
    
    # If the file already exists, append to it, otherwise create a new one
    if file_exists:
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, mode='w', header=True, index=False)
    print(f"Finished expriments ")
    

if __name__ == '__main__':
    main()