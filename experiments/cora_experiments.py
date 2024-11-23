"""
Comprehensive Cora extraction experiments.
Replicates Table 1 and Table 2 from the paper.
"""
import sys
sys.path.append('.')

import torch
import json
import numpy as np
from experiments.run_extraction import run_extraction_experiment
from src.data.dataset_loader import DatasetLoader
from src.utils.subgraph import SubgraphExtractor


def table1_experiments():
    """
    Replicate Table 1: Fidelity with varying subgraph sizes.
    10 samples per class on Cora.
    """
    print("\n" + "="*70)
    print("TABLE 1: Varying Subgraph Sizes (10 samples per class)")
    print("="*70 + "\n")
    
    # Find nodes with different subgraph sizes
    loader = DatasetLoader('Cora')
    data = loader.load()
    extractor = SubgraphExtractor(data)
    
    target_sizes = [15, 20, 25, 30, 40, 50, 60, 80, 100, 120]
    selected_nodes = []
    
    for node_idx in range(data.num_nodes):
        if len(selected_nodes) >= 10:
            break
        subgraph = extractor.extract_k_hop_subgraph(node_idx, num_hops=2)
        size = subgraph['num_nodes']
        if 10 <= size <= 150:
            selected_nodes.append({'node_idx': node_idx, 'size': size})
    
    results = []
    for node_info in selected_nodes[:10]:
        result = run_extraction_experiment(
            dataset_name='Cora',
            victim_model_path='models/victim_cora.pth',
            center_node=node_info['node_idx'],
            samples_per_class=10,
            epochs=200
        )
        results.append(result)
    
    # Print summary
    print("\nTable 1 Summary:")
    print(f"{'Node ID':<10} {'Subgraph Size':<15} {'Fidelity':<10}")
    print("-" * 35)
    for r in results:
        print(f"{r['center_node']:<10} {r['subgraph_size']:<15} {r['fidelity']:.4f}")
    
    avg_fidelity = np.mean([r['fidelity'] for r in results])
    print(f"\nAverage Fidelity: {avg_fidelity:.4f}")
    
    return results


def table2_experiments():
    """
    Replicate Table 2: Varying samples per class.
    Tests with 1, 3, 5, 10, 100 samples per class.
    """
    print("\n" + "="*70)
    print("TABLE 2: Varying Samples Per Class")
    print("="*70 + "\n")
    
    # Select 2 nodes with different sizes
    test_nodes = [675, 1956]  # As per paper
    samples_list = [1, 3, 5, 10, 100]
    
    all_results = []
    
    for node_idx in test_nodes:
        print(f"\nNode {node_idx}:")
        node_results = []
        
        for samples in samples_list:
            result = run_extraction_experiment(
                dataset_name='Cora',
                victim_model_path='models/victim_cora.pth',
                center_node=node_idx,
                samples_per_class=samples,
                epochs=200
            )
            node_results.append(result)
        
        all_results.extend(node_results)
        
        # Print summary for this node
        print(f"\nNode {node_idx} Summary:")
        print(f"{'Samples/Class':<15} {'Subgraph Size':<15} {'Fidelity':<10}")
        print("-" * 40)
        for r in node_results:
            print(f"{r['samples_per_class']:<15} {r['subgraph_size']:<15} {r['fidelity']:.4f}")
    
    return all_results


if __name__ == '__main__':
    # Run Table 1 experiments
    table1_results = table1_experiments()
    
    with open('experiments/results/table1_cora.json', 'w') as f:
        json.dump(table1_results, f, indent=2)
    
    # Run Table 2 experiments
    table2_results = table2_experiments()
    
    with open('experiments/results/table2_cora.json', 'w') as f:
        json.dump(table2_results, f, indent=2)
    
    print("\n" + "="*70)
    print("All experiments complete!")
    print("="*70)
