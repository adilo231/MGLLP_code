
import torch
from torch import Tensor

import torch.nn.functional as F
import torch_geometric.nn as nn
from torch_geometric.nn import SAGEConv,GCNConv


class GNN(torch.nn.Module):
    def __init__(self, input ,hidden_channels):
        super().__init__()

        self.conv1 = SAGEConv(input, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels//2)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.7, training=self.training)
        x = self.conv2(x, edge_index)
        return x
class GNN1(torch.nn.Module):
    def __init__(self, input ,hidden_channels):
        super().__init__()

        self.conv1 = SAGEConv(input, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels//2)
        self.fc1 = torch.nn.Linear(hidden_channels, 1)
        self.sigm = torch.nn.Sigmoid()

    def forward(self, x: Tensor, edge_index: Tensor,edge_label_index:Tensor) -> Tensor:

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.7, training=self.training)
        x = self.conv2(x, edge_index)
        edge_feat_1 = torch.index_select(
            x, 0, edge_label_index[0])
        edge_feat_2 = torch.index_select(
            x, 0, edge_label_index[1])

        x = torch.cat([edge_feat_1, edge_feat_2], dim=-1)
        x=self.fc1(x)
        x = self.sigm(x)
        return x

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
        x = F.dropout(x, p=0.7, training=self.training)
        x = self.relu(self.fc2(x))
        x = F.dropout(x, p=0.7, training=self.training)
        # Apply the final linear layer
        x = self.fc3(x)
        # Apply softmax activation to get probabilities
        x = self.sigm(x)
        return x

class Classifier1(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier1, self).__init__()

        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3 = torch.nn.Linear(hidden_dim//2, output_dim)
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x: Tensor,edge_label_index: Tensor) -> Tensor:
          # Pass the input through the MLP layers with ReLU activation
        edge_feat_1 = torch.index_select(
            x, 0, edge_label_index[0])
        edge_feat_2 = torch.index_select(
            x, 0, edge_label_index[1])
        x = torch.cat([edge_feat_1, edge_feat_2], dim=-1)
        # Pass the input through the MLP layers with ReLU activation
        x = self.relu(self.fc1(x))
        x = F.dropout(x, p=0.7, training=self.training)
        x = self.relu(self.fc2(x))
        x = F.dropout(x, p=0.7, training=self.training)
        # Apply the final linear layer
        x = self.fc3(x)
        # Apply softmax activation to get probabilities
        x = self.sigm(x)
        return x