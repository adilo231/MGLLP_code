import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
from torch_geometric.nn import GCNConv,SAGEConv
from torch_geometric.utils import train_test_split_edges, negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score
import torch_geometric.utils as utils
from tqdm import tqdm
import pickle
# Define the Graph Autoencoder (GAE)
import torch.nn.functional as F

class GAEModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.3,aggr="max"):
        super(GAEModel, self).__init__()
        # Define three GCN layers
        self.conv1 = SAGEConv(in_channels, hidden_channels,aggr=aggr)  # First GCN layer
        self.conv2 = SAGEConv(hidden_channels, hidden_channels,aggr=aggr)  # Second GCN layer
        self.conv3 = SAGEConv(hidden_channels, out_channels,aggr=aggr)  # Third GCN layer (final output)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout_rate)  # Dropout rate defined as 0.3 by default
    def decode(self, z, edge_index):
        # Decoder: compute the dot product between node embeddings
        edge_embeddings = z[edge_index[0]] * z[edge_index[1]]
        return edge_embeddings.sum(dim=1)
    def encode(self, x, edge_index):
        # Encoder: pass through multiple GCN layers with ReLU activations
        x = self.conv1(x, edge_index)
        x = F.relu(x)  # Apply activation function (ReLU)
        x = self.dropout(x)  # Apply dropout

        x = self.conv2(x, edge_index)
        x = F.relu(x)  # Apply activation function (ReLU)
        x = self.dropout(x)  # Apply dropout again

        z = self.conv3(x, edge_index)  # No activation on the final layer for embeddings
        return z
    
    def forward(self, x, edge_index):
        # Forward pass to get node embeddings
        z = self.encode(x, edge_index)
        return z


# Triplet Loss for link prediction
class TripletLossModel(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLossModel, self).__init__()
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin)
    
    def forward(self, anchor, positive, negative):
        # Compute the triplet loss
        loss = self.triplet_loss(anchor, positive, negative)
        return loss

# Prepare the dataset
def prepare_data(data):
    # Split edges for training, validation, and testing
    data = train_test_split_edges(data,val_ratio=0.1,test_ratio=0.1)
    return data

# Sample triplets for training
def sample_triplets(embeddings, pos_edge_index, neg_edge_index):
    # Positive samples (connected nodes)
    pos_samples = pos_edge_index[:, torch.randint(0, pos_edge_index.size(1), (1,))].squeeze(0)
    
    # Negative samples (disconnected nodes)
    neg_samples = neg_edge_index[:, torch.randint(0, neg_edge_index.size(1), (1,))].squeeze(0)
    
    # Return the anchor (node), positive (neighbor), and negative (non-neighbor) nodes
    return embeddings[pos_samples[0]], embeddings[pos_samples[1]], embeddings[neg_samples[1]]

def sample_neg_edges(num_neg_samples, num_nodes, edge_index):
    return negative_sampling(edge_index=edge_index, num_nodes=num_nodes, num_neg_samples=num_neg_samples)

# Function to compute ROC AUC and AP
def compute_metrics(z, pos_edge_index, neg_edge_index):
    # Positive edges
    pos_pred = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
    
    # Negative edges
    neg_pred = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
    
    # Labels (1 for positive edges, 0 for negative edges)
    labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))])
    
    # Predictions
    preds = torch.cat([pos_pred, neg_pred])
    
    # Compute ROC AUC and AP
    roc_auc = roc_auc_score(labels.cpu().detach().numpy(), preds.cpu().detach().numpy())
    ap_score = average_precision_score(labels.cpu().detach().numpy(), preds.cpu().detach().numpy())
    
    return roc_auc, ap_score


# Main training loop
def train(model, data, epochs=100, lr=0.001):
    # Early stopping parameters
    patience = 20  # Number of epochs with no improvement to wait
    best_val_metric = 0  # Best validation ROC AUC or AP score
    epochs_no_improve = 0  # Count of consecutive epochs with no improvement


    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    triplet_loss_fn = TripletLossModel(margin=1.0)

    for epoch in (range(epochs)):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        z = model(data.x, data.train_pos_edge_index)



        
        # Sample negative edges for link prediction
        neg_edge_index = negative_sampling(
            edge_index=data.train_pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.train_pos_edge_index.size(1),
        )

        # Sample triplets (anchor, positive, negative)
        anchor, positive, negative = sample_triplets(z, data.train_pos_edge_index, neg_edge_index)

        # Calculate triplet loss
        loss = triplet_loss_fn(anchor, positive, negative)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            
            roc_auc, ap_score = compute_metrics(z, data.train_pos_edge_index, neg_edge_index)
            print(f'Epoch {epoch}\t, Loss: {loss.item():.4f}, ROC AUC: {roc_auc:.4f}, AP: {ap_score:.4f}')
            val_neg_edge_index = sample_neg_edges(
            num_neg_samples=data.val_pos_edge_index.size(1), 
            num_nodes=data.num_nodes, 
            edge_index=data.train_pos_edge_index
                                    )
            val_roc_auc, val_ap_score = compute_metrics(z, data.val_pos_edge_index, val_neg_edge_index)
            print(f'\tValidation ROC AUC: {val_roc_auc:.4f}, AP: {val_ap_score:.4f}')
        # Early stopping logic: Track the best validation ROC AUC or AP
        if val_ap_score > best_val_metric:
            best_val_metric = val_ap_score
            epochs_no_improve = 0  # Reset counter
            print("Validation metric improved!")
        else:
            epochs_no_improve += 1
            

        # Stop training if no improvement for 'patience' number of epochs
        if epochs_no_improve >= patience:
            print(f"Early stopping after {epoch} epochs. Best Validation AP: {best_val_metric:.4f}")
            break

# Insert this at the end of the training loop (after the loop finishes)
def evaluate_test_set(model, data):
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.train_pos_edge_index)

        # Sample negative edges for the test set
        test_neg_edge_index = sample_neg_edges(
            num_neg_samples=data.test_pos_edge_index.size(1), 
            num_nodes=data.num_nodes, 
            edge_index=data.train_pos_edge_index
        )
        
        # Compute ROC AUC and AP on the test set
        test_roc_auc, test_ap_score = compute_metrics(z, data.test_pos_edge_index, test_neg_edge_index)
        print(f'Test ROC AUC: {test_roc_auc:.4f}, AP: {test_ap_score:.4f}')


# Example usage with PyTorch Geometric dataset
from torch_geometric.datasets import Planetoid


pickle_file_path =f"data/datasets/processed/BUP.pkl"
with open(pickle_file_path, "rb") as file:
        data = pickle.load( file)

data = prepare_data(data)
print(data.num_features)
# Initialize model
in_channels = data.num_features

hidden_channels= 256
out_channels = 32  # Embedding size 
gae_model = GAEModel(in_channels,hidden_channels, out_channels)

# Train model
train(gae_model, data)

# After training is complete, call the test evaluation
evaluate_test_set(gae_model, data)
