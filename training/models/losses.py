import torch
import numpy as np
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
    
class YOLOInspiredGlucoseLoss(nn.Module):
   """
   Custom loss function inspired by YOLO combining:
   1. Binary Cross Entropy for range prediction
   2. MSE for value prediction within range
   """
   def __init__(self, alpha: float = 1.0):
       super().__init__()
       self.alpha = alpha
       self.bce = nn.BCELoss()
       self.mse = nn.MSELoss()

   def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
       """
       Args:
           predictions: Model predictions [batch_size, num_ranges]
           targets: Target values [batch_size, num_ranges]
           Each vector contains zeros except one position with normalized value
       """
       # Binary Cross Entropy for range prediction
       # Convert targets to binary (1 for non-zero values, 0 for zeros)
       target_ranges = (targets > 0).float()
       range_loss = self.bce(predictions, target_ranges)

       # MSE for value prediction, but only for the correct range
       # Create mask for non-zero target positions
       value_mask = targets > 0
       
       if torch.any(value_mask):
           # Calculate MSE only for positions where target is non-zero
           value_loss = self.mse(
               predictions[value_mask],
               targets[value_mask]
           )
       else:
           value_loss = torch.tensor(0.0)

       # Combine losses
       total_loss = range_loss + self.alpha * value_loss

       return total_loss
   

def calculate_range_metrics(outputs: torch.Tensor, targets: torch.Tensor) -> tuple:
    """Calculate range-based accuracy and recall metrics.
    
    Args:
        outputs: Model predictions [batch_size, num_ranges]
        targets: Target values [batch_size, num_ranges]
        
    Returns:
        tuple: (accuracy, recall, mse)
    """
    # Get predicted range (max probability)
    pred_ranges = torch.argmax(outputs, dim=1)
    true_ranges = torch.argmax(targets, dim=1)
    
    # Calculate accuracy
    accuracy = (pred_ranges == true_ranges).float().mean().item()
    
    # Calculate recall for each range
    recalls = []
    for range_idx in range(outputs.shape[1]):
        true_positives = ((pred_ranges == range_idx) & (true_ranges == range_idx)).sum().item()
        total_actual = (true_ranges == range_idx).sum().item()
        recall = true_positives / total_actual if total_actual > 0 else 0.0
        recalls.append(recall)
    
    # Average recall across ranges
    mean_recall = np.mean(recalls)
    
    # Calculate MSE
    mse = torch.nn.functional.mse_loss(outputs, targets).item()
    
    return accuracy, mean_recall, mse