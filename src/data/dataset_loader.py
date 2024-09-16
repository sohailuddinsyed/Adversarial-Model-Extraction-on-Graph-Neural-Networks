import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
import os


class DatasetLoader:
    """Load and preprocess citation network datasets (Cora, Pubmed)."""
    
    def __init__(self, dataset_name, root='data'):
        """
        Initialize dataset loader.
        
        Args:
            dataset_name: Name of dataset ('Cora' or 'Pubmed')
            root: Root directory for dataset storage
        """
        self.dataset_name = dataset_name
        self.root = root
        self.dataset = None
        self.data = None
        
    def load(self):
        """Load the dataset."""
        self.dataset = Planetoid(root=self.root, name=self.dataset_name)
        self.data = self.dataset[0]
        
        # Convert to undirected graph as per paper
        self.data.edge_index = to_undirected(self.data.edge_index)
        
        return self.data
    
    def get_train_mask(self, num_train_per_class):
        """
        Create training mask with specified samples per class.
        
        Args:
            num_train_per_class: Number of training samples per class
            
        Returns:
            Boolean mask for training nodes
        """
        num_nodes = self.data.num_nodes
        num_classes = self.dataset.num_classes
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        for c in range(num_classes):
            class_indices = (self.data.y == c).nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(class_indices))
            train_indices = class_indices[perm[:num_train_per_class]]
            train_mask[train_indices] = True
            
        return train_mask
    
    def get_dataset_info(self):
        """Get dataset statistics."""
        if self.data is None:
            self.load()
            
        return {
            'num_nodes': self.data.num_nodes,
            'num_edges': self.data.edge_index.shape[1] // 2,
            'num_features': self.dataset.num_features,
            'num_classes': self.dataset.num_classes
        }
