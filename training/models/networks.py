import torch
import torch.nn as nn
import logging
from typing import Dict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActivationFactory:
    """Factory class for activation functions"""
    @staticmethod
    def get_activation(name: str, **kwargs) -> nn.Module:
        activations = {
            'leaky_relu': lambda: nn.LeakyReLU(**kwargs),
            'relu': lambda: nn.ReLU(),
            'elu': lambda: nn.ELU(**kwargs),
            'selu': lambda: nn.SELU(),
            'gelu': lambda: nn.GELU()
        }
        return activations.get(name.lower(), lambda: nn.LeakyReLU(**kwargs))()

class HybridFirstBlock(nn.Module):
    """Hybrid first block containing parallel dense and conv1d paths"""
    def __init__(self, input_size: int, hidden_size: int, kernel_size: int = 3, 
                 activation: str = 'leaky_relu', dropout_rate: float = 0.2, 
                 leaky_relu_slope: float = 0.01):
        super().__init__()

        # Calculate conv output size
        conv_hidden_size = 128

        # Activation function setup
        activation_params = {'negative_slope': leaky_relu_slope} if activation == 'leaky_relu' else {}
        activation_fn = ActivationFactory.get_activation(activation, **activation_params)

        # Dense path
        self.dense_path = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation_fn,
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate)
        )

        # Conv1D path
        self.conv_path = nn.Sequential(
            nn.Conv1d(1, conv_hidden_size, kernel_size + 2, padding='same'),
            activation_fn,
            nn.BatchNorm1d(conv_hidden_size),
            nn.Dropout(dropout_rate),

            # nn.Conv1d(conv_hidden_size, conv_hidden_size, kernel_size, padding='same'),
            # activation_fn,
            # nn.BatchNorm1d(conv_hidden_size),
            # nn.Dropout(dropout_rate),

            nn.Flatten()
        )

        # Output projection
        total_concat_size = hidden_size + (input_size * conv_hidden_size)
        self.output_projection = nn.Linear(total_concat_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dense_output = self.dense_path(x)
        conv_input = x.unsqueeze(1)
        conv_output = self.conv_path(conv_input)
        combined = torch.cat([dense_output, conv_output], dim=1)
        return self.output_projection(combined)

class HybridNetwork(nn.Module):
    """Neural Network with hybrid first block and standard fully connected layers"""
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 num_layers: int,
                 hidden_size: int,
                 dropout_rate: float = 0.2,
                 leaky_relu_slope: float = 0.01,
                 activation: str = 'leaky_relu',
                 conv_kernel_size: int = 3):
        super().__init__()

        self.architecture = {
            'input_size': input_size,
            'output_size': output_size,
            'num_layers': num_layers,
            'hidden_size': hidden_size,
            'dropout_rate': dropout_rate,
            'leaky_relu_slope': leaky_relu_slope,
            'activation': activation,
            'conv_kernel_size': conv_kernel_size
        }

        # First hybrid block
        self.first_block = HybridFirstBlock(
            input_size=input_size,
            hidden_size=hidden_size,
            kernel_size=conv_kernel_size,
            activation=activation,
            dropout_rate=dropout_rate,
            leaky_relu_slope=leaky_relu_slope
        )

        # Activation setup for subsequent layers
        activation_params = {'negative_slope': leaky_relu_slope} if activation == 'leaky_relu' else {}

        # Subsequent layers
        layers = []
        for i in range(1, num_layers):
            current_dropout = dropout_rate * (1 - i/num_layers)
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                ActivationFactory.get_activation(activation, **activation_params),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(current_dropout)
            ])

        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))

        self.subsequent_layers = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using He initialization"""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_normal_(
                module.weight,
                a=self.architecture['leaky_relu_slope'],
                nonlinearity='leaky_relu'
            )
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_block(x)
        return self.subsequent_layers(x)

    def get_info(self) -> Dict:
        """Return model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': self.architecture,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }