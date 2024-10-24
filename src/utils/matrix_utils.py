import torch
import scipy.sparse as sp
import numpy as np


def normalize_adjacency(adj):
    """
    Normalize adjacency matrix for GCN.
    A_norm = D^(-1/2) * A * D^(-1/2)
    
    Args:
        adj: Adjacency matrix
        
    Returns:
        Normalized adjacency matrix
    """
    # Add self-loops
    adj = adj + torch.eye(adj.shape[0])
    
    # Compute degree matrix
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    
    # Normalize
    adj_normalized = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    
    return adj_normalized


def sparse_to_dense(sparse_adj, num_nodes):
    """Convert sparse adjacency to dense."""
    dense = torch.zeros(num_nodes, num_nodes)
    indices = sparse_adj._indices()
    values = sparse_adj._values()
    dense[indices[0], indices[1]] = values
    return dense


def dense_to_sparse(dense_adj):
    """Convert dense adjacency to sparse."""
    indices = dense_adj.nonzero().t()
    values = dense_adj[indices[0], indices[1]]
    return torch.sparse_coo_tensor(indices, values, dense_adj.shape)
