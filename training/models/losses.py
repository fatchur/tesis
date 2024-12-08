import torch
import torch.nn as nn
from typing import List, Optional

class MedianBalancedMSELoss(nn.Module):
    """
    Custom loss function that combines MSE with a penalty for predictions
    clustering around median values.
    """
    def __init__(self, beta: float = 0.3, window_size: float = 0.2):
        super().__init__()
        self.beta = beta
        self.window_size = window_size
        self.mse = nn.MSELoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Calculate basic MSE loss
        mse_loss = self.mse(predictions, targets)

        # Calculate median of targets
        median = torch.median(targets)

        # Calculate the range of target values
        value_range = torch.max(targets) - torch.min(targets)
        window = value_range * self.window_size

        # Create a mask for predictions near median
        near_median_mask = torch.abs(predictions - median) < window

        # Calculate penalty for predictions clustering around median
        median_penalty = torch.mean(
            torch.exp(-torch.abs(predictions[near_median_mask] - median) / window)
            if torch.any(near_median_mask) else torch.tensor(0.0)
        )

        return mse_loss + self.beta * median_penalty

class RangeAwareMSELoss(nn.Module):
    """
    Custom loss function that applies higher weights to samples
    far from the median.
    """
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        squared_errors = (predictions - targets) ** 2
        median = torch.median(targets)
        distances_from_median = torch.abs(targets - median)
        max_distance = torch.max(distances_from_median)
        normalized_distances = distances_from_median / max_distance
        weights = 1.0 + self.alpha * normalized_distances
        return torch.mean(weights * squared_errors)

class QuantileBalancedMSELoss(nn.Module):
    """
    Custom loss function that balances MSE across different quantiles
    of the target distribution.
    """
    def __init__(self, num_quantiles: int = 5, 
                 quantile_weights: Optional[List[float]] = None):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.quantile_weights = (
            torch.tensor(quantile_weights) if quantile_weights
            else torch.ones(num_quantiles) / num_quantiles
        )

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Calculate quantile boundaries
        quantiles = torch.tensor([
            torch.quantile(targets, q)
            for q in torch.linspace(0, 1, self.num_quantiles + 1)
        ])

        total_loss = torch.tensor(0.0)

        # Calculate weighted loss for each quantile
        for i in range(self.num_quantiles):
            mask = (targets >= quantiles[i]) & (targets < quantiles[i + 1])
            if torch.any(mask):
                quantile_loss = torch.mean((predictions[mask] - targets[mask]) ** 2)
                total_loss += self.quantile_weights[i] * quantile_loss

        return total_loss