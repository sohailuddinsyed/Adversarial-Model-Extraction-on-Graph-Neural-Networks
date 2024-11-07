"""
Main script to run extraction experiments.
"""
import sys
sys.path.append('.')

import torch
import numpy as np
from src.data.dataset_loader import DatasetLoader
from src.utils.subgraph import SubgraphExtractor
from src.extraction import ExtractionAlgorithm, VictimAPI, load_victim_model, FidelityMeasure


def run_extraction_experiment(dataset_name, victim_model_path, center_node, 
                              samples_per_class, epochs=200):
    """
    Run a single extraction experiment.
    
    Args:
        dataset_name: 'Cora' or 'Pubmed'
        victim_model_path: Path to victim model
        center_node: Center node for subgraph
        samples_per_class: Number of samples per class
        epochs: Training epochs
        
    Returns:
        Dictionary with results
    """
    print(f"\n{'='*60}")
    print(f"Extraction Experiment: {dataset_name}")
    print(f"Center Node: {center_node}, Samples per class: {samples_per_class}")
    print(f"{'='*60}\n")
    
    # Load dataset
    loader = DatasetLoader(dataset_name)
    data = loader.load()
    num_classes = loader.dataset.num_classes
    num_features = loader.dataset.num_features
    
    # Load victim model
    victim_model = load_victim_model(
        victim_model_path,
        num_features=num_features,
        num_classes=num_classes,
        hidden_dim=16
    )
    
    # Create victim API
    victim_api = VictimAPI(victim_model, data)
    
    # Extract subgraph
    extractor = SubgraphExtractor(data)
    subgraph = extractor.extract_k_hop_subgraph(center_node, num_hops=2)
    
    print(f"Subgraph size: {subgraph['num_nodes']} nodes")
    
    # Run extraction
    extraction_algo = ExtractionAlgorithm(
        victim_api=victim_api,
        data=data,
        num_classes=num_classes,
        hidden_dim=16
    )
    
    extracted_model = extraction_algo.extract(
        subgraph_info=subgraph,
        samples_per_class=samples_per_class,
        epochs=epochs,
        use_hard_labels=True,
        verbose=True
    )
    
    # Measure fidelity on full graph
    print("\nMeasuring fidelity...")
    victim_preds = victim_api.query(return_probs=False)
    
    extracted_model.eval()
    with torch.no_grad():
        extracted_preds = extracted_model(data.x, data.edge_index).argmax(dim=1)
    
    fidelity = FidelityMeasure.compute_fidelity(victim_preds, extracted_preds)
    
    # Measure accuracy
    victim_acc = FidelityMeasure.compute_accuracy(victim_preds, data.y)
    extracted_acc = FidelityMeasure.compute_accuracy(extracted_preds, data.y)
    
    results = {
        'dataset': dataset_name,
        'center_node': center_node,
        'subgraph_size': subgraph['num_nodes'],
        'samples_per_class': samples_per_class,
        'fidelity': fidelity,
        'victim_accuracy': victim_acc,
        'extracted_accuracy': extracted_acc,
        'num_queries': victim_api.get_query_count()
    }
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Fidelity: {fidelity:.4f}")
    print(f"  Victim Accuracy: {victim_acc:.4f}")
    print(f"  Extracted Accuracy: {extracted_acc:.4f}")
    print(f"  Total Queries: {results['num_queries']}")
    print(f"{'='*60}\n")
    
    return results


if __name__ == '__main__':
    # Test extraction on Cora
    results = run_extraction_experiment(
        dataset_name='Cora',
        victim_model_path='models/victim_cora.pth',
        center_node=100,
        samples_per_class=5,
        epochs=200
    )
