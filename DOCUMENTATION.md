# Adversarial Model Extraction on GNNs - Documentation

## Overview

This project implements the paper "Adversarial Model Extraction on Graph Neural Networks" by DeFazio and Ramesh (2019). It demonstrates how to extract a Graph Convolutional Network (GCN) model given only API access to a victim model and a small subgraph.

## Project Structure

```
.
├── src/
│   ├── data/              # Dataset loaders
│   ├── models/            # GCN model definitions
│   ├── extraction/        # Extraction algorithms
│   └── utils/             # Utility functions
├── experiments/           # Experiment scripts
├── scripts/              # Training scripts
└── models/               # Saved models
```

## Key Components

### 1. Victim Model Training
- Train GCN models on Cora and Pubmed datasets
- Models saved in `models/` directory

### 2. Extraction Algorithm (Algorithm 1)
- Extract model using 2-hop subgraph access
- Generate samples for each class
- Train extracted model on synthetic samples

### 3. Algorithm 2 (Optional)
- Approximate inaccessible nodes
- Improve extraction fidelity

### 4. Fidelity Measurement
- Measure agreement between victim and extracted models
- Fidelity = percentage of nodes with same predictions

## Usage

### Train Victim Models

```bash
python scripts/train_victim_cora.py
python scripts/train_victim_pubmed.py
```

### Run Extraction

```bash
python experiments/run_extraction.py
```

### Run Batch Experiments

```bash
python experiments/cora_experiments.py
python experiments/pubmed_experiments.py
```

## Key Parameters

- `samples_per_class`: Number of synthetic samples per class (10-100)
- `epochs`: Training epochs for extracted model (200)
- `hidden_dim`: Hidden layer dimension (16)
- `epsilon`: Noise parameter for feature sampling (0.0)

## Expected Results

According to the paper:
- Fidelity: ~80% with sufficient samples
- Subgraph size: 10-150 nodes
- Samples needed: 10-100 per class

## Implementation Notes

1. **Complete Graph**: Subgraph is made complete to outweigh influence of inaccessible nodes
2. **Hard Labels**: Using argmax of victim predictions works better than soft labels
3. **Block Diagonal**: Training samples combined into block diagonal matrix
4. **Feature Sampling**: Sample from multinomial distributions per class

## References

DeFazio, D., & Ramesh, A. (2019). Adversarial Model Extraction on Graph Neural Networks. arXiv preprint arXiv:1912.07721.
