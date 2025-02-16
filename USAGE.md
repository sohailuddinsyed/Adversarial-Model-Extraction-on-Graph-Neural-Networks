# Usage Examples

## Basic Usage

### Run with default settings (Cora, 10 samples)
```bash
python main.py
```

### Specify dataset and parameters
```bash
python main.py --dataset Cora --node 675 --samples 20 --epochs 300
```

### Run on Pubmed
```bash
python main.py --dataset Pubmed --node 3028 --samples 15
```

## Advanced Usage

### Run batch experiments
```bash
# Cora experiments (Table 1 & 2 from paper)
python experiments/cora_experiments.py

# Pubmed experiments
python experiments/pubmed_experiments.py
```

### Custom extraction
```python
from experiments.run_extraction import run_extraction_experiment

result = run_extraction_experiment(
    dataset_name='Cora',
    victim_model_path='models/victim_cora.pth',
    center_node=100,
    samples_per_class=10,
    epochs=200
)

print(f"Fidelity: {result['fidelity']:.4f}")
```

## Parameters

- `--dataset`: Dataset name ('Cora' or 'Pubmed')
- `--node`: Center node ID for subgraph extraction
- `--samples`: Number of samples per class (10-100 recommended)
- `--epochs`: Training epochs for extracted model (200-300)

## Expected Runtime

- Training victim model: ~2-3 minutes
- Single extraction (10 samples): ~3-5 minutes
- Batch experiments: ~30-60 minutes

## Tips for Best Results

1. **More samples = higher fidelity** (but more queries)
2. **Choose nodes with 20-80 neighbors** for best results
3. **Use 200-300 epochs** for stable training
4. **Cora**: 10-50 samples per class
5. **Pubmed**: 15-30 samples per class (fewer classes)
