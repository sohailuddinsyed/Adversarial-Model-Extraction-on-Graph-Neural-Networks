"""Visualize experiment results."""
import matplotlib.pyplot as plt
import json
import numpy as np

def plot_fidelity_vs_samples(results, output_file='fidelity_vs_samples.png'):
    """Plot fidelity vs samples per class."""
    samples = [r['samples_per_class'] for r in results]
    fidelities = [r['fidelity'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(samples, fidelities, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Samples per Class', fontsize=12)
    plt.ylabel('Fidelity', fontsize=12)
    plt.title('Extraction Fidelity vs Samples per Class', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

def plot_fidelity_vs_subgraph_size(results, output_file='fidelity_vs_size.png'):
    """Plot fidelity vs subgraph size."""
    sizes = [r['subgraph_size'] for r in results]
    fidelities = [r['fidelity'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(sizes, fidelities, s=100, alpha=0.6)
    plt.xlabel('Subgraph Size (nodes)', fontsize=12)
    plt.ylabel('Fidelity', fontsize=12)
    plt.title('Extraction Fidelity vs Subgraph Size', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
