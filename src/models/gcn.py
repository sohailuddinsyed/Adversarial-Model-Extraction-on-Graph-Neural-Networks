import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """2-layer Graph Convolutional Network as described in the paper."""
    
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5):
        """
        Initialize GCN model.
        
        Args:
            num_features: Number of input features
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(GCN, self).__init__()
        
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Log probabilities [num_nodes, num_classes]
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def predict_proba(self, x, edge_index):
        """
        Get probability distributions (for victim API).
        
        Args:
            x: Node features
            edge_index: Edge indices
            
        Returns:
            Probability distributions [num_nodes, num_classes]
        """
        self.eval()
        with torch.no_grad():
            log_probs = self.forward(x, edge_index)
            probs = torch.exp(log_probs)
        return probs
    
    def predict(self, x, edge_index):
        """
        Get predicted class labels.
        
        Args:
            x: Node features
            edge_index: Edge indices
            
        Returns:
            Predicted labels [num_nodes]
        """
        probs = self.predict_proba(x, edge_index)
        return probs.argmax(dim=1)
