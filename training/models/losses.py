
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score, mean_squared_error

def calculate_metrics(outputs: torch.Tensor, targets: torch.Tensor, 
                     threshold: float = 0.5) -> tuple:
    """
    Calculate accuracy, recall, and MSE for binary classification
    
    Args:
        outputs: Model predictions (logits or probabilities)
        targets: Ground truth labels
        threshold: Classification threshold for binary predictions
    
    Returns:
        tuple: (accuracy, recall, mse)
    """
    with torch.no_grad():
        # Convert to numpy for sklearn metrics
        if outputs.requires_grad:
            outputs_np = outputs.detach().cpu().numpy()
        else:
            outputs_np = outputs.cpu().numpy()
            
        if targets.requires_grad:
            targets_np = targets.detach().cpu().numpy()
        else:
            targets_np = targets.cpu().numpy()
        
        # Handle different output formats
        if outputs_np.shape[-1] > 1:  # Multi-class or multi-output
            # For multi-class: use argmax
            predictions = outputs_np.argmax(axis=1)
            targets_binary = targets_np.argmax(axis=1) if targets_np.ndim > 1 else targets_np
        else:  # Binary classification
            # Apply sigmoid if using BCEWithLogitsLoss
            if outputs_np.min() < 0 or outputs_np.max() > 1:
                probs = torch.sigmoid(torch.from_numpy(outputs_np)).numpy()
            else:
                probs = outputs_np
            
            predictions = (probs > threshold).astype(int).flatten()
            targets_binary = targets_np.flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(targets_binary, predictions)
        
        # Handle recall calculation (avoid warnings for single class)
        try:
            recall = recall_score(targets_binary, predictions, average='binary', zero_division=0)
        except:
            recall = 0.0
        
        # MSE between outputs and targets
        mse = mean_squared_error(targets_np.flatten(), outputs_np.flatten())
        
        return accuracy, recall, mse