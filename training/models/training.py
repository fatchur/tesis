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
from models.losses import QuantileBalancedMSELoss

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
    """Handles model training and evaluation"""
    def __init__(self, model: nn.Module, train_loader: DataLoader, 
                 val_loader: DataLoader, config: Dict, 
                 model_manager: ModelManager):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.model_manager = model_manager

        # Initialize loss function
        self.criterion = QuantileBalancedMSELoss(
            num_quantiles=5, 
            quantile_weights=[0.1, 0.08, 0.05, 0.08, 0.1]
        )

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
        """Train the model"""
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        num_epochs = self.config.get('epochs', 500)

        for epoch in range(num_epochs):
            # Training phase
            train_loss = self._train_epoch()
            
            # Validation phase
            val_loss = self._validate_epoch()

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
                # Original saving logic as fallback
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
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

            # Print progress
            print(
                f'Epoch {epoch+1}/{num_epochs} | '
                f'Train Loss: {train_loss:.6f} | '
                f'Val Loss: {val_loss:.6f} | '
                f'LR: {self.optimizer.param_groups[0]["lr"]:.6f} '
                f'{improvement_marker}'
            )

            # Early stopping check
            # if self.early_stopping(self.model, val_loss):
            #     print("Early stopping triggered")
            #     break

        return train_losses, val_losses

    def _train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_size = 0
        
        for inputs, targets in self.train_loader:
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
            total_loss += loss.item() * inputs.size(0)
            total_size += inputs.size(0)

        return total_loss / total_size

    def _validate_epoch(self) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_size = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                total_size += inputs.size(0)

        return total_loss / total_size

class ModelEvaluator:
    """Handles model evaluation and metrics calculation"""
    @staticmethod
    def evaluate_model(model, data_loader: DataLoader, 
                      scale_factor: float = 1.0):
        """Evaluate model performance"""
        model.eval()
        metrics = {
            'mse': [], 'rmse': [], 'mae': [], 'mape': [], 'r2': []
        }

        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = model(inputs)
                predictions = outputs.numpy() * scale_factor
                targets = targets.numpy() * scale_factor

                batch_size = len(inputs)
                mse = np.mean((targets - predictions) ** 2) / batch_size
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(targets - predictions)) / batch_size

                # Calculate MAPE
                with np.errstate(divide='ignore', invalid='ignore'):
                    mape = np.mean(np.abs((targets - predictions) / targets)) * 100
                    mape = np.nan_to_num(mape, nan=0.0, posinf=0.0, neginf=0.0) / batch_size

                # Calculate R²
                ss_res = np.sum((targets - predictions) ** 2)
                ss_tot = np.sum((targets - np.mean(targets)) ** 2)
                r2 = (1 - (ss_res / ss_tot)) / batch_size if ss_tot != 0 else 0

                metrics['mse'].append(mse)
                metrics['rmse'].append(rmse)
                metrics['mae'].append(mae)
                metrics['mape'].append(mape)
                metrics['r2'].append(r2)

        # Calculate final metrics
        final_metrics = {
            metric: np.mean(values) for metric, values in metrics.items()
        }

        # Add standard deviation
        final_metrics.update({
            f'{metric}_std': np.std(values)
            for metric, values in metrics.items()
        })

        # Add confidence intervals
        n_batches = len(metrics['mse'])
        final_metrics.update({
            f'{metric}_ci': 1.96 * np.std(values) / np.sqrt(n_batches)
            for metric, values in metrics.items()
        })

        return final_metrics

    @staticmethod
    def print_metrics(metrics) -> None:
        """Print metrics in a formatted way"""
        print("\nModel Evaluation Metrics:")
        print("=" * 50)
        
        main_metrics = ['mse', 'rmse', 'mae', 'mape', 'r2']
        for metric in main_metrics:
            value = metrics[metric]
            std = metrics[f'{metric}_std']
            ci = metrics[f'{metric}_ci']
            print(f"{metric.upper():6s}: {value:.4f} ± {std:.4f} (95% CI: ±{ci:.4f})")
        
        print("=" * 50)