import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
from models.losses import YOLOInspiredGlucoseLoss, calculate_range_metrics
from config.config import RANGES
from datetime import datetime

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience: int = 7, min_delta: float = 0, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = None
        self.best_weights = None
        self.should_stop = False

    def __call__(self, model: nn.Module, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            self.counter = 0
        return self.should_stop

class ModelManager:
    """Handles model saving and loading"""
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)

    def save_model(self, model: nn.Module, optimizer: optim.Optimizer, 
                  epoch: int, train_loss: float, val_loss: float, 
                  filename: str) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_architecture': model.architecture,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'timestamp': datetime.now().isoformat()
        }

        filepath = os.path.join(self.models_dir, filename)
        torch.save(checkpoint, filepath)

    def load_model(self, filename: str, model_class: nn.Module) -> Tuple[nn.Module, Dict]:
        """Load model from checkpoint"""
        filepath = os.path.join(self.models_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}")

        checkpoint = torch.load(filepath)
        architecture = checkpoint['model_architecture']

        model = model_class(**architecture)
        model.load_state_dict(checkpoint['model_state_dict'])

        return model, checkpoint

class Trainer:
    """Handles model training and evaluation with additional metrics"""
    def __init__(self, model: nn.Module, train_loader: DataLoader, 
                 val_loader: DataLoader, config: Dict, 
                 model_manager: ModelManager):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.model_manager = model_manager

        # Initialize loss function
        self.criterion = YOLOInspiredGlucoseLoss(alpha=0.1)

        # Initialize optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 0.0001)
        )

        # Initialize learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.get('early_stopping_factor', 0.95),
            patience=config.get('early_stopping_patience', 5)
        )

        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 100),
            min_delta=config.get('early_stopping_delta', 1e-4)
        )

    def train(self, model_filename: str) -> Tuple[List[float], List[float]]:
        """Train the model with extended metrics tracking"""
        train_losses = []
        val_losses = []
        best_train_loss = float('inf')
        num_epochs = self.config.get('epochs', 500)

        for epoch in range(num_epochs):
            # Training phase
            train_metrics = self._train_epoch()
            train_loss = train_metrics['loss']
            
            # Validation phase
            val_metrics = self._validate_epoch()
            val_loss = val_metrics['loss']

            # Store losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Save model if better (using custom save function if available)
            if hasattr(self, 'save_if_better'):
                improved = self.save_if_better(
                    self.model, 
                    self.optimizer,
                    epoch,
                    train_loss,
                    val_loss
                )
                improvement_marker = "***" if improved else ""
            else:
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    self.model_manager.save_model(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        filename=model_filename
                    )
                    improvement_marker = "***"
                else:
                    improvement_marker = ""

            # Print progress with extended metrics
            current_time = datetime.now()
            print(
                f'Epoch {epoch+1}/{num_epochs} | '
                f'Train Loss: {train_loss:.4f} | '
                f'Val Loss: {val_loss:.4f} | '
                f'Train Acc: {train_metrics["accuracy"]:.2%} | '
                f'Val Acc: {val_metrics["accuracy"]:.2%} | '
                f'Train Recall: {train_metrics["recall"]:.2%} | '
                f'Val Recall: {val_metrics["recall"]:.2%} | '
                f'Train MSE: {train_metrics["mse"]:.4f} | '
                f'Val MSE: {val_metrics["mse"]:.4f} | '
                f'LR: {self.optimizer.param_groups[0]["lr"]:.6f} | '
                f'{current_time.strftime("%H:%M:%S")}'
                f'{improvement_marker}'
            )

            # Early stopping check
            # if self.early_stopping(self.model, val_loss):
            #     print("Early stopping triggered")
            #     break

        return train_losses, val_losses

    def _train_epoch(self) -> Dict:
        """Train for one epoch and return extended metrics"""
        self.model.train()
        metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'recall': 0.0,
            'mse': 0.0
        }
        total_size = 0
        
        for inputs, targets in self.train_loader:
            batch_size = inputs.size(0)
            inputs = inputs.requires_grad_(True)
            targets = targets.requires_grad_(True)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )

            self.optimizer.step()

            # Calculate additional metrics
            accuracy, recall, mse = calculate_range_metrics(outputs, targets)
            
            # Update metrics (weighted by batch size)
            metrics['loss'] += loss.item() * batch_size
            metrics['accuracy'] += accuracy * batch_size
            metrics['recall'] += recall * batch_size
            metrics['mse'] += mse * batch_size
            total_size += batch_size

        # Calculate final metrics
        for key in metrics:
            metrics[key] /= total_size

        return metrics

    def _validate_epoch(self) -> Dict:
        """Validate for one epoch and return extended metrics"""
        self.model.eval()
        metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'recall': 0.0,
            'mse': 0.0
        }
        total_size = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                batch_size = inputs.size(0)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Calculate additional metrics
                accuracy, recall, mse = calculate_range_metrics(outputs, targets)
                
                # Update metrics (weighted by batch size)
                metrics['loss'] += loss.item() * batch_size
                metrics['accuracy'] += accuracy * batch_size
                metrics['recall'] += recall * batch_size
                metrics['mse'] += mse * batch_size
                total_size += batch_size

        # Calculate final metrics
        for key in metrics:
            metrics[key] /= total_size

        return metrics
    

class ModelEvaluator:
    """Handles comprehensive model evaluation and metrics calculation"""
    @staticmethod
    def evaluate_model(model: nn.Module, data_loader: DataLoader, 
                      scale_factor: float = 1.0) -> Dict:
        """Evaluate model performance with comprehensive metrics"""
        model.eval()
        metrics = {
            'mse': [], 'rmse': [], 'mae': [], 'mape': [], 'r2': [],
            'accuracy': [], 'recall': [], 'range_recalls': {i: [] for i in range(len(RANGES))}
        }

        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = model(inputs)
                
                # Calculate range-based metrics
                accuracy, recall, mse = calculate_range_metrics(outputs, targets)
                
                # Calculate per-range recall
                pred_ranges = torch.argmax(outputs, dim=1)
                true_ranges = torch.argmax(targets, dim=1)
                for range_idx in range(len(RANGES)):
                    true_positives = ((pred_ranges == range_idx) & (true_ranges == range_idx)).sum().item()
                    total_actual = (true_ranges == range_idx).sum().item()
                    range_recall = true_positives / total_actual if total_actual > 0 else 0.0
                    metrics['range_recalls'][range_idx].append(range_recall)

                # Store range-based metrics
                metrics['accuracy'].append(accuracy)
                metrics['recall'].append(recall)
                
                # Get raw predictions for regression metrics
                # Find the value within the predicted range using argmax and range bounds
                pred_ranges = torch.argmax(outputs, dim=1)
                pred_values = torch.zeros_like(pred_ranges, dtype=torch.float32)
                
                for i, pred_range in enumerate(pred_ranges):
                    range_start, range_end = RANGES[pred_range]
                    range_value = outputs[i, pred_range].item()  # normalized value within range
                    if range_end == float('inf'):
                        # For the last range (>300/SCALE)
                        actual_value = range_start + range_value * (1.0 - range_start)
                    else:
                        # For other ranges
                        actual_value = range_start + range_value * (range_end - range_start)
                    pred_values[i] = actual_value
                
                # Get true values
                true_values = torch.zeros_like(true_ranges, dtype=torch.float32)
                for i, true_range in enumerate(true_ranges):
                    range_start, range_end = RANGES[true_range]
                    range_value = targets[i, true_range].item()
                    if range_end == float('inf'):
                        actual_value = range_start + range_value * (1.0 - range_start)
                    else:
                        actual_value = range_start + range_value * (range_end - range_start)
                    true_values[i] = actual_value

                # Scale values back to original range
                predictions = pred_values.numpy() * scale_factor
                true_targets = true_values.numpy() * scale_factor

                # Calculate regression metrics
                batch_size = len(inputs)
                mse = np.mean((true_targets - predictions) ** 2) / batch_size
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(true_targets - predictions)) / batch_size

                # Calculate MAPE
                with np.errstate(divide='ignore', invalid='ignore'):
                    mape = np.mean(np.abs((true_targets - predictions) / true_targets)) * 100
                    mape = np.nan_to_num(mape, nan=0.0, posinf=0.0, neginf=0.0) / batch_size

                # Calculate R²
                ss_res = np.sum((true_targets - predictions) ** 2)
                ss_tot = np.sum((true_targets - np.mean(true_targets)) ** 2)
                r2 = (1 - (ss_res / ss_tot)) / batch_size if ss_tot != 0 else 0

                # Store metrics
                metrics['mse'].append(mse)
                metrics['rmse'].append(rmse)
                metrics['mae'].append(mae)
                metrics['mape'].append(mape)
                metrics['r2'].append(r2)

        # Calculate final metrics
        final_metrics = {
            metric: np.mean(values) 
            for metric, values in metrics.items() 
            if metric != 'range_recalls'
        }

        # Add per-range recall metrics
        for range_idx, recalls in metrics['range_recalls'].items():
            range_start, range_end = RANGES[range_idx]
            range_name = f"{range_start * scale_factor:.0f}-{range_end * scale_factor if range_end != float('inf') else 'inf'}"
            final_metrics[f'recall_range_{range_name}'] = np.mean(recalls)

        # Add standard deviation
        final_metrics.update({
            f'{metric}_std': np.std(values)
            for metric, values in metrics.items()
            if metric != 'range_recalls'
        })

        # Add confidence intervals
        n_batches = len(metrics['mse'])
        final_metrics.update({
            f'{metric}_ci': 1.96 * np.std(values) / np.sqrt(n_batches)
            for metric, values in metrics.items()
            if metric != 'range_recalls'
        })

        return final_metrics

    @staticmethod
    def print_metrics(metrics: Dict) -> None:
        """Print metrics in a formatted way"""
        print("\nModel Evaluation Metrics:")
        print("=" * 50)
        
        # Print main metrics
        main_metrics = ['accuracy', 'recall', 'mse', 'rmse', 'mae', 'mape', 'r2']
        for metric in main_metrics:
            value = metrics[metric]
            std = metrics[f'{metric}_std']
            ci = metrics[f'{metric}_ci']
            print(f"{metric.upper():6s}: {value:.4f} ± {std:.4f} (95% CI: ±{ci:.4f})")
        
        # Print per-range recalls
        print("\nPer-Range Recall:")
        for key in metrics.keys():
            if key.startswith('recall_range_'):
                range_name = key.replace('recall_range_', '')
                value = metrics[key]
                print(f"Range {range_name}: {value:.2%}")
        
        print("=" * 50)