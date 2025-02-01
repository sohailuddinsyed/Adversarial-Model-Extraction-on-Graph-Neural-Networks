# Adversarial Model Extraction on Graph Neural Networks

Implementation of the paper "Adversarial Model Extraction on Graph Neural Networks" by David DeFazio and Arti Ramesh (arXiv:1912.07721v1).

## Overview

This project implements a model extraction attack on Graph Neural Networks (GNNs), demonstrating how an adversary can steal a GCN model with only:
- API access to victim model predictions
- A small 2-hop subgraph (10-150 nodes)

**Key Result**: Achieves **71.8% fidelity** (vs paper's 80%) with 10 samples per class on Cora dataset.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Victim Models
```bash
python scripts/train_victim_cora.py
python scripts/train_victim_pubmed.py
```

### 3. Run Extraction Attack
```bash
python experiments/run_extraction.py
```

## Project Structure

```
.
├── src/
│   ├── data/              # Dataset loaders (Cora, Pubmed)
│   ├── models/            # GCN architecture
│   ├── extraction/        # Algorithm 1 & 2 implementation
│   └── utils/             # Graph utilities, feature sampling
├── experiments/           # Experiment scripts
│   ├── run_extraction.py      # Single extraction experiment
│   ├── cora_experiments.py    # Batch Cora experiments
│   └── pubmed_experiments.py  # Batch Pubmed experiments
├── scripts/               # Training scripts
└── models/               # Saved victim models
```

## Key Features

✓ **Algorithm 1**: Core extraction using subgraph sampling  
✓ **Algorithm 2**: Approximate inaccessible nodes  
✓ **Fidelity Measurement**: Compare victim vs extracted predictions  
✓ **Multiple Datasets**: Cora (7 classes) and Pubmed (3 classes)  
✓ **Configurable**: Samples per class, epochs, noise parameters  

## Results

| Dataset | Samples/Class | Our Fidelity | Paper Fidelity |
|---------|---------------|--------------|----------------|
| Cora    | 10            | **71.8%**    | ~80%           |
| Cora    | 50            | **75.0%**    | ~82%           |

See [RESULTS.md](RESULTS.md) for detailed analysis.

## How It Works

1. **Victim Training**: Train GCN on citation network
2. **Subgraph Access**: Extract 2-hop neighborhood around center node
3. **Sample Generation**: Create synthetic samples for each class
4. **Query Victim**: Get predictions for perturbed subgraphs
5. **Train Extraction**: Learn model that mimics victim predictions
6. **Measure Fidelity**: Test agreement on full graph

## Documentation

- [DOCUMENTATION.md](DOCUMENTATION.md) - Detailed usage guide
- [RESULTS.md](RESULTS.md) - Experimental results and analysis

## Citation

```
@article{defazio2019adversarial,
  title={Adversarial Model Extraction on Graph Neural Networks},
  author={DeFazio, David and Ramesh, Arti},
  journal={arXiv preprint arXiv:1912.07721},
  year={2019}
}
```

## License

This is a research implementation for educational purposes.
