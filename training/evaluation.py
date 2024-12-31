import torch
import os
from models.networks import HybridNetwork
from models.training import ModelManager, ModelEvaluator
from utils.data_processing import DataProcessor
from utils.visualization import Visualizer
from config.config import VERSION, SCALE, LOWER_THD, UPPER_THD


def main():
    # Setup paths
    print ("starting ...")
    base_dir = f'data/v{VERSION}'
    model_name = f'v_HYBRID_{VERSION}_{int(LOWER_THD * SCALE)}_{int(UPPER_THD * SCALE)}'
    model_path = os.path.join(base_dir, 'models', f"{model_name}.pth")
    print ("model path: {}".format(model_path))
    # Initialize model manager
    model_manager = ModelManager(base_dir)

    try:
        # Load model
        print(f"Loading model from {model_path}")
        model, checkpoint = model_manager.load_model(f"{model_name}.pth", HybridNetwork)
        print(f"Model loaded from epoch {checkpoint['epoch']}")

        # Load validation data
        data_processor = DataProcessor()
        val_feature, val_target, _, _ = data_processor.load_data()
        _, val_loader = data_processor.prepare_data(
            train_feature=val_feature,  # Dummy train data
            train_target=val_target,    # Dummy train data
            val_feature=val_feature,
            val_target=val_target,
            batch_size=512
        )

        # Make predictions
        model.eval()
        predictions = []
        targets = []
        with torch.no_grad():
            for inputs, target in val_loader:
                output = model(inputs)
                predictions.append(output)
                targets.append(target)

        # Concatenate predictions and targets
        predictions = torch.cat(predictions).numpy() * SCALE
        targets = torch.cat(targets).numpy() * SCALE

        # Create plots directory
        plots_dir = os.path.join(base_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        # Plot and save predictions vs actual values
        Visualizer.plot_predictions(
            predictions=predictions,
            targets=targets,
            save_path=os.path.join(plots_dir, f'{model_name}_evaluation2.png')
        )

        # Calculate and print metrics
        metrics = ModelEvaluator.evaluate_model(
            model=model,
            data_loader=val_loader,
            scale_factor=SCALE
        )
        ModelEvaluator.print_metrics(metrics)

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise


if __name__ == "__main__":
    main()