"""Pubmed extraction experiments."""
import sys
sys.path.append('.')

from experiments.run_extraction import run_extraction_experiment
import json

# Pubmed experiments (3 classes, so need more samples per class)
samples_list = [1, 7, 11, 23]
test_nodes = [3028, 12759]

results = []
for node_idx in test_nodes:
    for samples in samples_list:
        result = run_extraction_experiment(
            dataset_name='Pubmed',
            victim_model_path='models/victim_pubmed.pth',
            center_node=node_idx,
            samples_per_class=samples,
            epochs=200
        )
        results.append(result)

with open('experiments/results/pubmed_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Pubmed experiments complete!")
