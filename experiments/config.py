"""Configuration for experiments."""

# Model hyperparameters
HIDDEN_DIM = 16
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
DROPOUT = 0.5

# Extraction parameters
DEFAULT_SAMPLES_PER_CLASS = 10
DEFAULT_EPOCHS = 200
EPSILON = 0.0  # Noise parameter

# Subgraph constraints
MIN_SUBGRAPH_SIZE = 10
MAX_SUBGRAPH_SIZE = 150

# Dataset paths
CORA_VICTIM_PATH = 'models/victim_cora.pth'
PUBMED_VICTIM_PATH = 'models/victim_pubmed.pth'
