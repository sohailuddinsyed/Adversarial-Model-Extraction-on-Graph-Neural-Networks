"""Analyze and visualize experiment results."""
import json
import numpy as np

def analyze_results(filename):
    """Analyze results from JSON file."""
    with open(filename, 'r') as f:
        results = json.load(f)
    
    fidelities = [r['fidelity'] for r in results]
    
    print(f"\nResults from {filename}:")
    print(f"  Number of experiments: {len(results)}")
    print(f"  Average fidelity: {np.mean(fidelities):.4f}")
    print(f"  Std deviation: {np.std(fidelities):.4f}")
    print(f"  Min fidelity: {np.min(fidelities):.4f}")
    print(f"  Max fidelity: {np.max(fidelities):.4f}")
    
    return results

if __name__ == '__main__':
    print("="*60)
    print("Experiment Results Analysis")
    print("="*60)
