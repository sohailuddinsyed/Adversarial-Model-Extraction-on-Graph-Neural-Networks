"""
Main execution script for GNN Model Extraction.
Run complete extraction pipeline with single command.
"""
import sys
import argparse
from experiments.run_extraction import run_extraction_experiment


def main():
    parser = argparse.ArgumentParser(description='GNN Model Extraction')
    parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora', 'Pubmed'],
                       help='Dataset to use')
    parser.add_argument('--node', type=int, default=100,
                       help='Center node for subgraph extraction')
    parser.add_argument('--samples', type=int, default=10,
                       help='Samples per class')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Training epochs')
    
    args = parser.parse_args()
    
    # Set victim model path
    if args.dataset == 'Cora':
        victim_path = 'models/victim_cora.pth'
    else:
        victim_path = 'models/victim_pubmed.pth'
    
    print("\n" + "="*70)
    print("GNN MODEL EXTRACTION")
    print("="*70)
    print(f"\nDataset: {args.dataset}")
    print(f"Center Node: {args.node}")
    print(f"Samples per Class: {args.samples}")
    print(f"Training Epochs: {args.epochs}")
    print("\n" + "="*70 + "\n")
    
    # Run extraction
    result = run_extraction_experiment(
        dataset_name=args.dataset,
        victim_model_path=victim_path,
        center_node=args.node,
        samples_per_class=args.samples,
        epochs=args.epochs
    )
    
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)
    print(f"\nFidelity: {result['fidelity']:.4f}")
    print(f"Victim Accuracy: {result['victim_accuracy']:.4f}")
    print(f"Extracted Accuracy: {result['extracted_accuracy']:.4f}")
    print(f"Queries Used: {result['num_queries']}")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
