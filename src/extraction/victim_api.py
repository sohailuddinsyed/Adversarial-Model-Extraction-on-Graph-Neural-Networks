import torch
from src.models.gcn import GCN


class VictimAPI:
    """
    API interface to victim model.
    Simulates black-box access where adversary can only query predictions.
    """
    
    def __init__(self, model, data):
        """
        Initialize victim API.
        
        Args:
            model: Trained victim GCN model
            data: Full graph data
        """
        self.model = model
        self.data = data
        self.model.eval()
        self.query_count = 0
        
    def query(self, node_indices=None, return_probs=True):
        """
        Query victim model for predictions.
        
        Args:
            node_indices: Specific nodes to query (None = all nodes)
            return_probs: Return probability distributions (True) or labels (False)
            
        Returns:
            Predictions for queried nodes
        """
        self.query_count += 1
        
        with torch.no_grad():
            if return_probs:
                probs = self.model.predict_proba(self.data.x, self.data.edge_index)
            else:
                probs = self.model.predict(self.data.x, self.data.edge_index)
        
        if node_indices is not None:
            return probs[node_indices]
        return probs
    
    def query_subgraph(self, subgraph_data, return_probs=True):
        """
        Query victim on a modified subgraph.
        This simulates perturbing the graph and getting new predictions.
        
        Args:
            subgraph_data: Dictionary with 'x', 'edge_index', 'subset' (original indices)
            return_probs: Return probability distributions or labels
            
        Returns:
            Predictions for subgraph nodes
        """
        self.query_count += 1
        
        # Create temporary modified graph
        modified_x = self.data.x.clone()
        modified_edge_index = self.data.edge_index.clone()
        
        # Update features for subgraph nodes
        subset = subgraph_data['subset']
        modified_x[subset] = subgraph_data['x']
        
        # Update edges if provided
        if 'edge_index' in subgraph_data and subgraph_data['edge_index'] is not None:
            # This is simplified - in practice would need to carefully merge edges
            pass
        
        with torch.no_grad():
            if return_probs:
                probs = self.model.predict_proba(modified_x, modified_edge_index)
            else:
                probs = self.model.predict(modified_x, modified_edge_index)
        
        return probs[subset]
    
    def get_query_count(self):
        """Get number of queries made to victim."""
        return self.query_count
    
    def reset_query_count(self):
        """Reset query counter."""
        self.query_count = 0


def load_victim_model(model_path, num_features, num_classes, hidden_dim=16):
    """
    Load a trained victim model.
    
    Args:
        model_path: Path to saved model
        num_features: Number of input features
        num_classes: Number of output classes
        hidden_dim: Hidden dimension
        
    Returns:
        Loaded GCN model
    """
    model = GCN(num_features, hidden_dim, num_classes)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model
