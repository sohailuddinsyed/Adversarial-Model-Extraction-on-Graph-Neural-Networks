import torch
import numpy as np
from torch_geometric.utils import k_hop_subgraph


class SubgraphExtractor:
    """Extract k-hop subgraphs from the main graph."""
    
    def __init__(self, data):
        """
        Initialize subgraph extractor.
        
        Args:
            data: PyTorch Geometric data object
        """
        self.data = data
        self.num_nodes = data.num_nodes
        
    def extract_k_hop_subgraph(self, center_node, num_hops=2):
        """
        Extract k-hop subgraph around a center node.
        
        Args:
            center_node: Center node index
            num_hops: Number of hops (default 2 as per paper)
            
        Returns:
            Dictionary containing subgraph information
        """
        # Get k-hop subgraph
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            center_node,
            num_hops,
            self.data.edge_index,
            relabel_nodes=True,
            num_nodes=self.num_nodes
        )
        
        # Extract node features for subgraph
        x_subgraph = self.data.x[subset]
        
        # Get labels if available
        y_subgraph = self.data.y[subset] if hasattr(self.data, 'y') else None
        
        return {
            'subset': subset,  # Original node indices
            'edge_index': edge_index,  # Relabeled edge indices
            'x': x_subgraph,  # Node features
            'y': y_subgraph,  # Node labels
            'center_mapping': mapping,  # Index of center node in subgraph
            'num_nodes': len(subset)
        }
    
    def find_valid_center_nodes(self, min_size=10, max_size=150):
        """
        Find nodes with 2-hop neighborhoods in valid size range.
        As per paper: subgraph should have 10-150 nodes.
        
        Args:
            min_size: Minimum subgraph size
            max_size: Maximum subgraph size
            
        Returns:
            List of valid center node indices
        """
        valid_nodes = []
        
        for node_idx in range(self.num_nodes):
            subgraph = self.extract_k_hop_subgraph(node_idx, num_hops=2)
            size = subgraph['num_nodes']
            
            if min_size <= size <= max_size:
                valid_nodes.append({
                    'node_idx': node_idx,
                    'subgraph_size': size
                })
        
        return valid_nodes
    
    def get_node_degrees(self, subset):
        """
        Get degrees of nodes in subset.
        
        Args:
            subset: Node indices
            
        Returns:
            Tensor of node degrees
        """
        edge_index = self.data.edge_index
        degrees = torch.zeros(len(subset), dtype=torch.long)
        
        for i, node in enumerate(subset):
            degree = (edge_index[0] == node).sum().item()
            degrees[i] = degree
            
        return degrees
