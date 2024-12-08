import torch
import numpy as np

# Constants
FEATURE_NUM = 8000
UPPER_THD = 300/800
LOWER_THD = 0/800
SCALE = 800.
VERSION = 5
SELECT_FEATURE = False

# Training configuration
TRAINING_CONFIG = {
    'batch_size': 2,
    'val_batch_size': 2,
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'epochs': 500,
    'early_stopping_patience': 5,
    'early_stopping_delta': 1e-4,
    'num_layers': 1,
    'hidden_size': 512,
    'dropout_rate': 0.2,
    'leaky_relu_slope': 0.01,
    'activation': 'leaky_relu'
}

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)