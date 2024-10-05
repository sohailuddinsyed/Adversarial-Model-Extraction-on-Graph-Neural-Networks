import torch
import numpy as np
from scipy.sparse import coo_matrix


def edge_index_to_adjacency(edge_index, num_nodes):
    """
    Convert edge_index to dense adjacency matrix.
    
    Args:
        edge_index: Edge indices [2, num_edges]
        num_nodes: Number of nodes
        
    Returns:
        Dense adjacency matrix [num_nodes, num_nodes]
    """
    adj = torch.zeros(num_nodes, num_nodes)
    adj[edge_index[0], edge_index[1]] = 1
    return adj


def adjacency_to_edge_index(adj):
    """
    Convert dense adjacency matrix to edge_index.
    
    Args:
        adj: Dense adjacency matrix [num_nodes, num_nodes]
        
    Returns:
        Edge indices [2, num_edges]
    """
    edge_index = adj.nonzero().t()
    return edge_index


def create_complete_graph(num_nodes):
    """
    Create a complete graph (all nodes connected).
    As per Algorithm 1 line 17: make subgraph complete.
    
    Args:
        num_nodes: Number of nodes
        
    Returns:
        Edge index for complete graph
    """
    # Create all possible edges (excluding self-loops)
    rows = []
    cols = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)
    
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    return edge_index


def create_block_diagonal_adjacency(adj_list):
    """
    Create block diagonal matrix from list of adjacency matrices.
    As per Algorithm 1 line 11: combine samples into block diagonal.
    
    Args:
        adj_list: List of adjacency matrices
        
    Returns:
        Block diagonal adjacency matrix
    """
    if not adj_list:
        return torch.tensor([])
    
    # Get sizes
    sizes = [adj.shape[0] for adj in adj_list]
    total_size = sum(sizes)
    
    # Create block diagonal matrix
    block_diag = torch.zeros(total_size, total_size)
    
    offset = 0
    for adj in adj_list:
        size = adj.shape[0]
        block_diag[offset:offset+size, offset:offset+size] = adj
        offset += size
    
    return block_diag


def get_neighbors_at_distance(edge_index, node, distance, num_nodes):
    """
    Get all neighbors at exactly 'distance' hops from node.
    
    Args:
        edge_index: Edge indices
        node: Center node
        distance: Number of hops
        num_nodes: Total number of nodes
        
    Returns:
        Set of neighbor indices at given distance
    """
    if distance == 0:
        return {node}
    
    # Build adjacency list
    adj_list = [[] for _ in range(num_nodes)]
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        adj_list[src].append(dst)
    
    # BFS to find neighbors at exact distance
    current_level = {node}
    visited = {node}
    
    for _ in range(distance):
        next_level = set()
        for n in current_level:
            for neighbor in adj_list[n]:
                if neighbor not in visited:
                    next_level.add(neighbor)
                    visited.add(neighbor)
        current_level = next_level
    
    return current_level
