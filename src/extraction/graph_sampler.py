import torch
import numpy as np
from src.utils.graph_utils import create_complete_graph
from src.utils.feature_distribution import FeatureDistribution


class GraphSampler:
    """
    Generate graph samples for extraction training.
    Implements GenerateSample function from Algorithm 1.
    """
    
    def __init__(self, data, num_classes):
        """
        Initialize graph sampler.
        
        Args:
            data: PyTorch Geometric data object
            num_classes: Number of classes in dataset
        """
        self.data = data
        self.num_classes = num_classes
        self.feat_dist = FeatureDistribution(data)
        
    def generate_sample(self, subgraph_adjacency, class_label, epsilon=0.0):
        """
        Generate a sample for given class as per Algorithm 1 lines 16-24.
        
        Args:
            subgraph_adjacency: Original subgraph adjacency matrix
            class_label: Class to generate sample for
            epsilon: Noise probability for feature sampling
            
        Returns:
            Dictionary with adjacency and features
        """
        num_nodes = subgraph_adjacency.shape[0]
        
        # Line 17: Create complete graph with same nodes as As
        adjacency_complete = self._create_complete_adjacency(num_nodes)
        
        # Lines 18-23: Generate features for each node
        features = self.feat_dist.generate_sample_features(
            class_label=class_label,
            num_nodes=num_nodes,
            epsilon=epsilon
        )
        
        return {
            'adjacency': adjacency_complete,
            'features': features,
            'num_nodes': num_nodes,
            'class_label': class_label
        }
    
    def _create_complete_adjacency(self, num_nodes):
        """
        Create complete graph adjacency matrix.
        
        Args:
            num_nodes: Number of nodes
            
        Returns:
            Dense adjacency matrix for complete graph
        """
        # Create complete graph (all nodes connected)
        adjacency = torch.ones(num_nodes, num_nodes)
        # Remove self-loops
        adjacency.fill_diagonal_(0)
        return adjacency
    
    def generate_samples_all_classes(self, subgraph_adjacency, samples_per_class, epsilon=0.0):
        """
        Generate samples for all classes as per Algorithm 1 lines 4-10.
        
        Args:
            subgraph_adjacency: Original subgraph adjacency
            samples_per_class: Number of samples per class (n in paper)
            epsilon: Noise probability
            
        Returns:
            List of sample dictionaries
        """
        samples = []
        
        # For each iteration (line 4)
        for i in range(samples_per_class):
            # For each class (line 5)
            for c in range(self.num_classes):
                sample = self.generate_sample(
                    subgraph_adjacency=subgraph_adjacency,
                    class_label=c,
                    epsilon=epsilon
                )
                samples.append(sample)
        
        return samples
    
    def create_training_batch(self, samples, original_subgraph=None):
        """
        Create training batch from samples.
        Combines samples into block diagonal matrix as per Algorithm 1 line 11-12.
        
        Args:
            samples: List of sample dictionaries
            original_subgraph: Optional original subgraph to include
            
        Returns:
            Dictionary with combined adjacency, features, and metadata
        """
        all_adjacencies = []
        all_features = []
        all_labels = []
        node_offsets = []
        
        # Include original subgraph if provided (lines 1-3)
        if original_subgraph is not None:
            all_adjacencies.append(original_subgraph['adjacency'])
            all_features.append(original_subgraph['features'])
            all_labels.append(original_subgraph['labels'])
            node_offsets.append(0)
        
        # Add all samples (lines 6-8)
        offset = 0
        if original_subgraph is not None:
            offset = original_subgraph['adjacency'].shape[0]
        
        for sample in samples:
            all_adjacencies.append(sample['adjacency'])
            all_features.append(sample['features'])
            node_offsets.append(offset)
            offset += sample['num_nodes']
        
        # Create block diagonal adjacency matrix (line 11)
        block_adjacency = self._create_block_diagonal(all_adjacencies)
        
        # Concatenate features (line 12)
        combined_features = torch.cat(all_features, dim=0)
        
        return {
            'adjacency': block_adjacency,
            'features': combined_features,
            'labels': all_labels if original_subgraph is not None else None,
            'node_offsets': node_offsets,
            'num_samples': len(samples)
        }
    
    def _create_block_diagonal(self, adjacency_list):
        """
        Create block diagonal matrix from list of adjacency matrices.
        
        Args:
            adjacency_list: List of adjacency matrices
            
        Returns:
            Block diagonal adjacency matrix
        """
        if not adjacency_list:
            return torch.tensor([])
        
        sizes = [adj.shape[0] for adj in adjacency_list]
        total_size = sum(sizes)
        
        block_diag = torch.zeros(total_size, total_size)
        
        offset = 0
        for adj in adjacency_list:
            size = adj.shape[0]
            block_diag[offset:offset+size, offset:offset+size] = adj
            offset += size
        
        return block_diag
