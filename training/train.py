from utils.data_processing import DataProcessor
from models.networks import HybridNetwork
from models.training import ModelManager, Trainer, ModelEvaluator
from config.config import TRAINING_CONFIG, VERSION, SCALE, LOWER_THD, UPPER_THD
import os
import torch

BASE_DIR = "/Users/asani/Documents/Documents/tesis/training"

def main():
    """Main execution function"""
    # Setup paths
    print("Starting training pipeline...")
    base_dir = f'{BASE_DIR}/data/v{VERSION}'
    model_name = f'v_HYBRID_{VERSION}_{int(LOWER_THD * SCALE)}_{int(UPPER_THD * SCALE)}'
    model_path = os.path.join(base_dir, 'models', f"{model_name}.pth")

    try:
        # Load and prepare data
        print("Loading and preparing data...")
        data_processor = DataProcessor()
        train_feature, train_target, val_feature, val_target, test_feature, test_target = data_processor.load_data(BASE_DIR)
        train_loader, val_loader, test_loader = data_processor.prepare_data(
            train_feature=train_feature,
            train_target=train_target,
            val_feature=val_feature,
            val_target=val_target,
            test_feature=test_feature,
            test_target=test_target,
            batch_size=TRAINING_CONFIG['batch_size'],
            val_batch_size=TRAINING_CONFIG['val_batch_size'],
            test_batch_size=None
        )

        print (test_loader)
        print("Data processing completed successfully")

        # Initialize model manager
        model_manager = ModelManager(base_dir)

        # Try to load existing model if it exists
        model = None
        best_val_loss = float('inf')
        previous_checkpoint = None
        
        try:
            if os.path.exists(model_path):
                print(f"Found existing model at {model_path}")
                model, previous_checkpoint = model_manager.load_model(
                    f"{model_name}.pth",
                    HybridNetwork
                )
                best_val_loss = previous_checkpoint['val_loss']
                print(f"Loaded model from epoch {previous_checkpoint['epoch']} with best validation loss: {best_val_loss:.6f}")
                
                # Print model architecture info
                model_info = model.get_info()
                print("\nModel Architecture:")
                for key, value in model_info['architecture'].items():
                    print(f"{key}: {value}")
                print(f"Total parameters: {model_info['total_parameters']:,}")
                print(f"Trainable parameters: {model_info['trainable_parameters']:,}\n")
        except Exception as e:
            print(f"Could not load existing model: {str(e)}")
            model = None

        # Create new model if no existing model was loaded
        if model is None:
            print("Initializing new model...")
            model = HybridNetwork(
                input_size=train_feature.shape[1],
                output_size=1,
                num_layers=TRAINING_CONFIG['num_layers'],
                hidden_size=TRAINING_CONFIG['hidden_size'],
                dropout_rate=TRAINING_CONFIG['dropout_rate'],
                leaky_relu_slope=TRAINING_CONFIG['leaky_relu_slope'],
                activation=TRAINING_CONFIG['activation']
            )
            print("New model initialized successfully")

            # Print model architecture info
            model_info = model.get_info()
            print("\nModel Architecture:")
            for key, value in model_info['architecture'].items():
                print(f"{key}: {value}")
            print(f"Total parameters: {model_info['total_parameters']:,}")
            print(f"Trainable parameters: {model_info['trainable_parameters']:,}\n")

        # Define custom save function with comparison logic
        def save_if_better(current_model, optimizer, epoch, train_loss, val_loss):
            nonlocal best_val_loss
            if val_loss < best_val_loss:
                print(f"New best validation loss: {val_loss:.6f} (previous: {best_val_loss:.6f})")
                best_val_loss = val_loss
                model_manager.save_model(
                    model=current_model,
                    optimizer=optimizer,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    filename=f"{model_name}.pth"
                )
                return True
            return False

        # Initialize trainer with custom save function
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=TRAINING_CONFIG,
            model_manager=model_manager
        )
        # Attach custom save function to trainer
        trainer.save_if_better = save_if_better

        # Train model
        print("\nStarting training...")
        if previous_checkpoint:
            print(f"Previous best validation loss to beat: {best_val_loss:.6f}")
        train_losses, val_losses = trainer.train(f"{model_name}.pth")
        print("Training completed successfully")

        # Evaluate model
        print("\nEvaluating model...")
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model(
            model=trainer.model,
            data_loader=val_loader,
            scale_factor=SCALE
        )
        evaluator.print_metrics(metrics)

        # Print final best loss comparison
        if previous_checkpoint:
            print(f"\nOriginal best validation loss: {previous_checkpoint['val_loss']:.6f}")
        print(f"Final best validation loss: {best_val_loss:.6f}")

    except Exception as e:
        print(f"Error during training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()