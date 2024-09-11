# Adversarial Model Extraction on Graph Neural Networks

Implementation of the paper "Adversarial Model Extraction on Graph Neural Networks" by David DeFazio and Arti Ramesh (arXiv:1912.07721v1).

## Overview

This project implements a model extraction attack on Graph Neural Networks (GNNs), specifically targeting Graph Convolutional Networks (GCNs) trained on citation network datasets.

## Setup

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── src/                    # Source code
├── data/                   # Dataset storage
├── experiments/            # Experiment scripts and results
└── models/                 # Model definitions
```

## Reference

```
@article{defazio2019adversarial,
  title={Adversarial Model Extraction on Graph Neural Networks},
  author={DeFazio, David and Ramesh, Arti},
  journal={arXiv preprint arXiv:1912.07721},
  year={2019}
}
```
