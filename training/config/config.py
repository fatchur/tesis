import torch
import numpy as np

# Constants
FEATURE_NUM = 8000
UPPER_THD = 800/800
LOWER_THD = 0/800
SCALE = 800.
VERSION = 7
SELECT_FEATURE = False
RANGES = [
            (0, 100/SCALE),
            (100/SCALE, 200/SCALE),
            (200/SCALE, 300/SCALE),
            (300/SCALE, float('inf'))
        ]

# Training configuration
TRAINING_CONFIG = {
    'batch_size': 256,
    'val_batch_size': 512,
    'learning_rate': 0.0001,
    'weight_decay': 0.00005,
    'epochs': 2000,
    'early_stopping_patience': 10,
    'early_stopping_factor': 0.995,
    'early_stopping_delta': 1e-4,
    'num_layers': 2,
    'hidden_size': 32,
    'dropout_rate': 0.3,
    'leaky_relu_slope': 0.01,
    'activation': 'leaky_relu',
    'loss_alpha': 0.3,
    'loss_betha': 0.2
}

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)