import torch
import numpy as np


class FeatureDistribution:
    """
    Compute and sample from feature distributions per class.
    As per paper: F and M are multinomial distributions for features.
    """
    
    def __init__(self, data):
        """
        Initialize feature distribution computer.
        
        Args:
            data: PyTorch Geometric data object with x (features) and y (labels)
        """
        self.data = data
        self.num_classes = int(data.y.max().item()) + 1
        self.num_features = data.x.shape[1]
        
        # Compute distributions
        self.feature_distributions = {}  # F_c in paper
        self.num_features_distributions = {}  # M_c in paper
        
        self._compute_distributions()
    
    def _compute_distributions(self):
        """
        Compute F and M distributions for each class.
        F_c: probability distribution over features for class c
        M_c: probability distribution over number of features for class c
        """
        for c in range(self.num_classes):
            # Get nodes of this class
            class_mask = (self.data.y == c)
            class_features = self.data.x[class_mask]
            
            if class_features.shape[0] == 0:
                continue
            
            # Compute feature frequency (F_c)
            # For binary features: count how often each feature appears
            feature_counts = class_features.sum(dim=0)
            feature_probs = feature_counts / feature_counts.sum()
            self.feature_distributions[c] = feature_probs
            
            # Compute number of features distribution (M_c)
            # Count how many features each node has
            num_features_per_node = class_features.sum(dim=1)
            unique_counts, counts = torch.unique(num_features_per_node, return_counts=True)
            
            # Create probability distribution
            num_features_probs = torch.zeros(int(num_features_per_node.max().item()) + 1)
            for count, freq in zip(unique_counts, counts):
                num_features_probs[int(count.item())] = freq.item()
            
            num_features_probs = num_features_probs / num_features_probs.sum()
            self.num_features_distributions[c] = num_features_probs
    
    def sample_features(self, class_label, epsilon=0.0):
        """
        Sample features for a node of given class.
        As per Algorithm 1 GenerateSample function.
        
        Args:
            class_label: Class to sample from
            epsilon: Noise probability (sample random feature instead)
            
        Returns:
            Binary feature vector
        """
        if class_label not in self.feature_distributions:
            # Fallback: random features
            num_features = np.random.randint(1, 50)
            features = torch.zeros(self.num_features)
            indices = torch.randperm(self.num_features)[:num_features]
            features[indices] = 1
            return features
        
        # Sample number of features from M_c
        num_features_probs = self.num_features_distributions[class_label]
        num_features = torch.multinomial(num_features_probs, 1).item()
        
        if num_features == 0:
            num_features = 1
        
        # Sample features from F_c
        feature_probs = self.feature_distributions[class_label]
        features = torch.zeros(self.num_features)
        
        for _ in range(int(num_features)):
            if np.random.random() < epsilon:
                # Add noise: sample random feature
                feature_idx = np.random.randint(0, self.num_features)
            else:
                # Sample from distribution
                feature_idx = torch.multinomial(feature_probs, 1).item()
            
            features[feature_idx] = 1
        
        return features
    
    def generate_sample_features(self, class_label, num_nodes, epsilon=0.0):
        """
        Generate features for multiple nodes of same class.
        
        Args:
            class_label: Class to sample from
            num_nodes: Number of nodes to generate
            epsilon: Noise probability
            
        Returns:
            Feature matrix [num_nodes, num_features]
        """
        features = torch.zeros(num_nodes, self.num_features)
        
        for i in range(num_nodes):
            features[i] = self.sample_features(class_label, epsilon)
        
        return features
    
    def get_distribution_stats(self):
        """Get statistics about the distributions."""
        stats = {}
        for c in range(self.num_classes):
            if c in self.feature_distributions:
                stats[c] = {
                    'top_features': self.feature_distributions[c].topk(5).indices.tolist(),
                    'avg_num_features': (self.num_features_distributions[c] * 
                                        torch.arange(len(self.num_features_distributions[c]))).sum().item()
                }
        return stats
