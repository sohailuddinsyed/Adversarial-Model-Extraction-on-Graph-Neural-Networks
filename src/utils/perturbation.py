import torch
import numpy as np


class GraphPerturbation:
    """Utilities for perturbing graphs during extraction."""
    
    @staticmethod
    def perturb_subgraph_features(data, subset, new_features):
        """
        Perturb features of nodes in subgraph.
        
        Args:
            data: Original graph data
            subset: Node indices to perturb
            new_features: New feature matrix for subset
            
        Returns:
            Modified data object
        """
        modified_data = data.clone()
        modified_data.x[subset] = new_features
        return modified_data
    
    @staticmethod
    def add_edges_to_subgraph(edge_index, subset, make_complete=True):
        """
        Add edges within subgraph.
        
        Args:
            edge_index: Original edge indices
            subset: Node indices of subgraph
            make_complete: Make subgraph complete
            
        Returns:
            Modified edge index
        """
        if make_complete:
            # Create all possible edges within subset
            new_edges = []
            for i in subset:
                for j in subset:
                    if i != j:
                        new_edges.append([i.item(), j.item()])
            
            if new_edges:
                new_edge_tensor = torch.tensor(new_edges, dtype=torch.long).t()
                # Combine with original edges
                edge_index = torch.cat([edge_index, new_edge_tensor], dim=1)
        
        return edge_index
    
    @staticmethod
    def compute_degree_distribution(edge_index, num_nodes):
        """
        Compute degree distribution of graph.
        
        Args:
            edge_index: Edge indices
            num_nodes: Number of nodes
            
        Returns:
            Degree for each node
        """
        degrees = torch.zeros(num_nodes, dtype=torch.long)
        for i in range(num_nodes):
            degrees[i] = (edge_index[0] == i).sum()
        return degrees
    
    @staticmethod
    def measure_perturbation_impact(original_edge_index, perturbed_edge_index, num_nodes):
        """
        Measure impact of perturbation on graph structure.
        
        Args:
            original_edge_index: Original edges
            perturbed_edge_index: Perturbed edges
            num_nodes: Number of nodes
            
        Returns:
            Dictionary with perturbation statistics
        """
        orig_degrees = GraphPerturbation.compute_degree_distribution(original_edge_index, num_nodes)
        pert_degrees = GraphPerturbation.compute_degree_distribution(perturbed_edge_index, num_nodes)
        
        return {
            'original_edges': original_edge_index.shape[1],
            'perturbed_edges': perturbed_edge_index.shape[1],
            'edges_added': perturbed_edge_index.shape[1] - original_edge_index.shape[1],
            'avg_degree_change': (pert_degrees.float() - orig_degrees.float()).abs().mean().item()
        }
