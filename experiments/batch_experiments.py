"""
Run batch extraction experiments with varying parameters.
"""
import sys
sys.path.append('.')

import torch
import json
import numpy as np
from datetime import datetime
from src.data.dataset_loader import DatasetLoader
from src.utils.subgraph import SubgraphExtractor
from experiments.run_extraction import run_extraction_experiment


def find_valid_nodes(dataset_name, min_size=10, max_size=150, num_nodes=10):
    """Find valid center nodes for experiments."""
    loader = DatasetLoader(dataset_name)
    data = loader.load()
    extractor = SubgraphExtractor(data)
    
    print(f"Finding valid nodes for {dataset_name}...")
    valid_nodes = []
    
    for node_idx in range(data.num_nodes):
        if len(valid_nodes) >= num_nodes:
            break
        
        subgraph = extractor.extract_k_hop_subgraph(node_idx, num_hops=2)
        size = subgraph['num_nodes']
        
        if min_size <= size <= max_size:
            valid_nodes.append({
                'node_idx': node_idx,
                'subgraph_size': size
            })
    
    return valid_nodes


def run_varying_samples_experiment(dataset_name, victim_model_path, node_idx, 
                                   samples_list=[1, 3, 5, 10]):
    """Run experiments with varying samples per class."""
    results = []
    
    for samples in samples_list:
        result = run_extraction_experiment(
            dataset_name=dataset_name,
            victim_model_path=victim_model_path,
            center_node=node_idx,
            samples_per_class=samples,
            epochs=200
        )
        results.append(result)
    
    return results


def run_varying_nodes_experiment(dataset_name, victim_model_path, 
                                 node_list, samples_per_class=10):
    """Run experiments with varying center nodes."""
    results = []
    
    for node_info in node_list:
        result = run_extraction_experiment(
            dataset_name=dataset_name,
            victim_model_path=victim_model_path,
            center_node=node_info['node_idx'],
            samples_per_class=samples_per_class,
            epochs=200
        )
        results.append(result)
    
    return results


def save_results(results, filename):
    """Save results to JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")


if __name__ == '__main__':
    # Find valid nodes for Cora
    cora_nodes = find_valid_nodes('Cora', num_nodes=5)
    print(f"Found {len(cora_nodes)} valid nodes for Cora")
    
    # Run experiments
    print("\nRunning Cora experiments...")
    cora_results = run_varying_nodes_experiment(
        dataset_name='Cora',
        victim_model_path='models/victim_cora.pth',
        node_list=cora_nodes[:3],
        samples_per_class=10
    )
    
    save_results(cora_results, 'experiments/results/cora_results.json')
