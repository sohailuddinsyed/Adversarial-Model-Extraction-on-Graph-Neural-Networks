import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from src.models.gcn import GCN
from src.extraction.graph_sampler import GraphSampler
from src.extraction.victim_api import VictimAPI
from src.utils.graph_utils import adjacency_to_edge_index


class ExtractionAlgorithm:
    """
    Implementation of Algorithm 1 from the paper.
    Extracts a GCN model given API access to victim and a 2-hop subgraph.
    """
    
    def __init__(self, victim_api, data, num_classes, hidden_dim=16):
        """
        Initialize extraction algorithm.
        
        Args:
            victim_api: VictimAPI object for querying victim
            data: Full graph data
            num_classes: Number of classes
            hidden_dim: Hidden dimension for extracted model
        """
        self.victim_api = victim_api
        self.data = data
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Initialize graph sampler
        self.sampler = GraphSampler(data, num_classes)
        
        # Extracted model (to be trained)
        self.extracted_model = None
        
    def extract(self, subgraph_info, samples_per_class, epochs=200, lr=0.01, 
                epsilon=0.0, use_hard_labels=True, verbose=True):
        """
        Main extraction algorithm (Algorithm 1).
        
        Args:
            subgraph_info: Dictionary with subgraph information
            samples_per_class: Number of samples per class (n in paper)
            epochs: Training epochs for extracted model
            lr: Learning rate
            epsilon: Noise parameter for feature sampling
            use_hard_labels: Use hard labels (argmax) vs soft labels (probabilities)
            verbose: Print progress
            
        Returns:
            Trained extracted model
        """
        if verbose:
            print(f"Starting extraction with {samples_per_class} samples per class...")
        
        # Lines 1-3: Initialize with original subgraph
        SA = []  # List of adjacency matrices
        SX = []  # List of feature matrices
        SL = []  # List of labels
        
        # Get original subgraph adjacency and features
        original_adj = self._get_subgraph_adjacency(subgraph_info)
        original_features = subgraph_info['x']
        
        # Query victim for original subgraph labels (line 3)
        original_labels = self.victim_api.query(
            node_indices=subgraph_info['subset'],
            return_probs=True
        )
        
        SA.append(original_adj)
        SX.append(original_features)
        SL.append(original_labels)
        
        if verbose:
            print(f"Original subgraph: {original_features.shape[0]} nodes")
        
        # Lines 4-10: Generate samples for each class
        print("Generating samples...")
        for i in range(samples_per_class):
            for c in range(self.num_classes):
                # Line 6: Generate sample (Ac, Xc)
                sample = self.sampler.generate_sample(
                    subgraph_adjacency=original_adj,
                    class_label=c,
                    epsilon=epsilon
                )
                
                # Line 7-8: Add to sets
                SA.append(sample['adjacency'])
                SX.append(sample['features'])
                
                # Line 9-10: Perturb graph and query victim
                # Create modified subgraph data for querying
                modified_subgraph = {
                    'subset': subgraph_info['subset'],
                    'x': sample['features'],
                    'edge_index': None  # Using complete graph
                }
                
                # Query victim with perturbed subgraph
                sample_labels = self.victim_api.query_subgraph(
                    modified_subgraph,
                    return_probs=True
                )
                
                SL.append(sample_labels)
        
        total_samples = len(SA)
        if verbose:
            print(f"Generated {total_samples} total samples")
        
        # Line 11: Create block diagonal adjacency matrix
        if verbose:
            print("Creating block diagonal matrix...")
        AG = self._create_block_diagonal_adjacency(SA)
        
        # Line 12: Concatenate features
        XG = torch.cat(SX, dim=0)
        
        # Concatenate labels
        LG = torch.cat(SL, dim=0)
        
        # Convert to hard labels if specified
        if use_hard_labels:
            LG = LG.argmax(dim=1)
        
        if verbose:
            print(f"Training data: {XG.shape[0]} nodes, {AG.shape[1]} edges")
        
        # Line 13: Train extracted GCN
        if verbose:
            print("Training extracted model...")
        self.extracted_model = self._train_extracted_model(
            AG, XG, LG, epochs, lr, use_hard_labels, verbose
        )
        
        if verbose:
            print("Extraction complete!")
        return self.extracted_model
    
    def _get_subgraph_adjacency(self, subgraph_info):
        """Get adjacency matrix for subgraph."""
        num_nodes = subgraph_info['num_nodes']
        edge_index = subgraph_info['edge_index']
        
        # Convert edge_index to dense adjacency
        adj = torch.zeros(num_nodes, num_nodes)
        if edge_index.shape[1] > 0:
            adj[edge_index[0], edge_index[1]] = 1
        
        return adj
    
    def _create_block_diagonal_adjacency(self, adjacency_list):
        """Create block diagonal matrix from list of adjacencies."""
        sizes = [adj.shape[0] for adj in adjacency_list]
        total_size = sum(sizes)
        
        block_diag = torch.zeros(total_size, total_size)
        
        offset = 0
        for adj in adjacency_list:
            size = adj.shape[0]
            block_diag[offset:offset+size, offset:offset+size] = adj
            offset += size
        
        return block_diag
    
    def _train_extracted_model(self, adjacency, features, labels, epochs, lr, use_hard_labels, verbose=True):
        """Train the extracted GCN model (line 13)."""
        # Convert adjacency to edge_index
        edge_index = adjacency_to_edge_index(adjacency)
        edge_index = to_undirected(edge_index)
        
        # Initialize model
        model = GCN(
            num_features=features.shape[1],
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            out = model(features, edge_index)
            
            if use_hard_labels:
                # Cross-entropy loss with hard labels
                loss = F.nll_loss(out, labels)
            else:
                # KL divergence loss with soft labels
                log_probs = F.log_softmax(out, dim=1)
                loss = F.kl_div(log_probs, labels, reduction='batchmean')
            
            loss.backward()
            optimizer.step()
            
            if verbose and (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        model.eval()
        return model
