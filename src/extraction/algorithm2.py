import torch
import numpy as np


class ApproximateInaccessibleNodes:
    """
    Implementation of Algorithm 2 from the paper.
    Approximates inaccessible nodes (3-4 hops away) for better extraction.
    """
    
    def __init__(self, rho=2):
        """
        Initialize Algorithm 2.
        
        Args:
            rho: Multiplier for number of nodes to add (default 2 as per paper)
        """
        self.rho = rho
    
    def approximate_nodes(self, subgraph_sample, victim_labels, center_idx, num_classes):
        """
        Approximate inaccessible nodes for a subgraph sample.
        Algorithm 2 from paper.
        
        Args:
            subgraph_sample: Dictionary with subgraph info
            victim_labels: Victim predictions for subgraph [num_nodes, num_classes]
            center_idx: Index of center node in subgraph
            num_classes: Number of classes
            
        Returns:
            Modified subgraph with approximated nodes added
        """
        # Line 2: Get labels from victim
        L = victim_labels
        
        # Line 3: Get center node label
        L_center = L[center_idx]
        
        # Get adjacency and features
        adjacency = subgraph_sample['adjacency']
        features = subgraph_sample['features']
        num_nodes = adjacency.shape[0]
        
        # Lists to store new nodes
        new_adjacencies = []
        new_features = []
        
        # Line 4: For each non-center node
        for n in range(num_nodes):
            if n == center_idx:
                continue
            
            # Line 5: Compute difference D_n = L_n - L_center
            D_n = L[n] - L_center
            
            # Line 6-8: For each class where D_n[c] > 0
            for c in range(num_classes):
                if D_n[c] > 0:
                    # Line 8: Compute number of nodes to add
                    degree_n = adjacency[n].sum().item()
                    num_nodes_to_add = int(D_n[c].item() * self.rho * degree_n)
                    
                    if num_nodes_to_add > 0:
                        # Add nodes of class c connected to node n
                        for _ in range(num_nodes_to_add):
                            # Create new node feature (sample from class c distribution)
                            new_node_feature = self._sample_node_feature(
                                features, c, num_classes
                            )
                            new_features.append(new_node_feature)
        
        # Line 9-10: Add new nodes to subgraph
        if new_features:
            new_features_tensor = torch.stack(new_features)
            augmented_features = torch.cat([features, new_features_tensor], dim=0)
            
            # Expand adjacency matrix
            num_new_nodes = len(new_features)
            new_size = num_nodes + num_new_nodes
            augmented_adjacency = torch.zeros(new_size, new_size)
            augmented_adjacency[:num_nodes, :num_nodes] = adjacency
            
            # Connect new nodes (simplified: connect to random existing nodes)
            for i in range(num_new_nodes):
                # Connect to a few random nodes
                num_connections = min(3, num_nodes)
                connections = torch.randperm(num_nodes)[:num_connections]
                for conn in connections:
                    augmented_adjacency[num_nodes + i, conn] = 1
                    augmented_adjacency[conn, num_nodes + i] = 1
        else:
            augmented_features = features
            augmented_adjacency = adjacency
        
        return {
            'adjacency': augmented_adjacency,
            'features': augmented_features,
            'num_nodes': augmented_adjacency.shape[0],
            'num_added_nodes': len(new_features) if new_features else 0
        }
    
    def _sample_node_feature(self, existing_features, class_label, num_classes):
        """
        Sample a feature vector for a new node.
        Simplified: sample from existing features.
        
        Args:
            existing_features: Existing feature matrix
            class_label: Class to sample for
            num_classes: Number of classes
            
        Returns:
            Feature vector for new node
        """
        # Simplified: randomly sample from existing features
        idx = torch.randint(0, existing_features.shape[0], (1,)).item()
        return existing_features[idx].clone()
