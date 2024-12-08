import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional
import os
import pandas as pd

class Visualizer:
    """Class for handling all visualization tasks"""
    
    @staticmethod
    def plot_training_history(
        train_losses: List[float],
        val_losses: List[float],
        save_path: Optional[str] = None,
        figsize: tuple = (10, 6)
    ) -> None:
        """
        Plot training and validation loss history
        
        Parameters:
        -----------
        train_losses : List[float]
            List of training losses
        val_losses : List[float]
            List of validation losses
        save_path : Optional[str]
            Path to save the plot. If None, display the plot
        figsize : tuple
            Figure size (width, height)
        """
        plt.figure(figsize=figsize)
        plt.plot(train_losses, label='Training Loss', color='blue', alpha=0.7)
        plt.plot(val_losses, label='Validation Loss', color='red', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_predictions(
        predictions: np.ndarray,
        targets: np.ndarray,
        save_path: Optional[str] = None,
        figsize: tuple = (10, 6)
    ) -> None:
        """
        Plot predictions vs actual values with additional statistics
        
        Parameters:
        -----------
        predictions : np.ndarray
            Model predictions
        targets : np.ndarray
            Actual target values
        save_path : Optional[str]
            Path to save the plot. If None, display the plot
        figsize : tuple
            Figure size (width, height)
        """
        plt.figure(figsize=figsize)
        
        # Create scatter plot
        plt.scatter(targets, predictions, alpha=0.5, color='blue', label='Predictions')
        
        # Plot perfect prediction line
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 
                'r--', lw=2, label='Perfect Prediction')
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(targets.flatten(), predictions.flatten())[0, 1]
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Predictions vs Actual Values (Correlation: {correlation:.3f})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Add text box with statistics
        stats_text = (f'Mean Error: {np.mean(predictions - targets):.3f}\n'
                     f'Std Error: {np.std(predictions - targets):.3f}\n'
                     f'Correlation: {correlation:.3f}')
        plt.text(0.05, 0.95, stats_text,
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_feature_importance(
        feature_names: List[str],
        importance_values: np.ndarray,
        top_n: int = 20,
        save_path: Optional[str] = None,
        figsize: tuple = (12, 8)
    ) -> None:
        """
        Plot feature importance
        
        Parameters:
        -----------
        feature_names : List[str]
            List of feature names
        importance_values : np.ndarray
            Array of importance values
        top_n : int
            Number of top features to display
        save_path : Optional[str]
            Path to save the plot. If None, display the plot
        figsize : tuple
            Figure size (width, height)
        """
        # Create DataFrame for feature importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_values
        })
        
        # Sort and get top N features
        importance_df = importance_df.sort_values(
            by='Importance', 
            ascending=True
        ).tail(top_n)
        
        plt.figure(figsize=figsize)
        
        # Create horizontal bar plot
        sns.barplot(
            data=importance_df,
            y='Feature',
            x='Importance',
            palette='viridis'
        )
        
        plt.title(f'Top {top_n} Most Important Features')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature Name')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_error_distribution(
        predictions: np.ndarray,
        targets: np.ndarray,
        save_path: Optional[str] = None,
        figsize: tuple = (10, 6)
    ) -> None:
        """
        Plot distribution of prediction errors
        
        Parameters:
        -----------
        predictions : np.ndarray
            Model predictions
        targets : np.ndarray
            Actual target values
        save_path : Optional[str]
            Path to save the plot. If None, display the plot
        figsize : tuple
            Figure size (width, height)
        """
        errors = predictions - targets
        
        plt.figure(figsize=figsize)
        
        # Create histogram of errors
        sns.histplot(errors, kde=True)
        
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Prediction Error')
        plt.ylabel('Count')
        plt.title('Distribution of Prediction Errors')
        
        # Add statistics text box
        stats_text = (f'Mean Error: {np.mean(errors):.3f}\n'
                     f'Std Error: {np.std(errors):.3f}\n'
                     f'Median Error: {np.median(errors):.3f}')
        plt.text(0.95, 0.95, stats_text,
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top',
                horizontalalignment='right')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def create_training_report(
        base_dir: str,
        model_name: str,
        train_losses: List[float],
        val_losses: List[float],
        predictions: np.ndarray,
        targets: np.ndarray,
        feature_names: Optional[List[str]] = None,
        feature_importance: Optional[np.ndarray] = None
    ) -> None:
        """
        Create a complete training report with multiple plots
        
        Parameters:
        -----------
        base_dir : str
            Base directory to save plots
        model_name : str
            Name of the model for file naming
        train_losses : List[float]
            Training loss history
        val_losses : List[float]
            Validation loss history
        predictions : np.ndarray
            Model predictions
        targets : np.ndarray
            Actual target values
        feature_names : Optional[List[str]]
            List of feature names for feature importance plot
        feature_importance : Optional[np.ndarray]
            Feature importance values
        """
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(base_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot training history
        Visualizer.plot_training_history(
            train_losses,
            val_losses,
            save_path=os.path.join(plots_dir, f'{model_name}_training_history.png')
        )
        
        # Plot predictions
        Visualizer.plot_predictions(
            predictions,
            targets,
            save_path=os.path.join(plots_dir, f'{model_name}_predictions.png')
        )
        
        # Plot error distribution
        Visualizer.plot_error_distribution(
            predictions,
            targets,
            save_path=os.path.join(plots_dir, f'{model_name}_error_distribution.png')
        )
        
        # Plot feature importance if provided
        if feature_names is not None and feature_importance is not None:
            Visualizer.plot_feature_importance(
                feature_names,
                feature_importance,
                save_path=os.path.join(plots_dir, f'{model_name}_feature_importance.png')
            )