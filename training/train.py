from utils.data_processing import DataProcessor
from models.networks import HybridNetwork
from models.training import ModelManager, Trainer, ModelEvaluator
from config.config import TRAINING_CONFIG, VERSION, SCALE, LOWER_THD, UPPER_THD

def main():
    """Main execution function"""
    # Setup paths
    print ("starting ...")
    base_dir = f'data/v{VERSION}'
    model_name = f'v_HYBRID_{VERSION}_{int(LOWER_THD * SCALE)}_{int(UPPER_THD * SCALE)}'

    try:
        # Load and prepare data
        data_processor = DataProcessor()
        train_feature, train_target, val_feature, val_target = data_processor.load_data()
        train_loader, val_loader = data_processor.prepare_data(
            train_feature=train_feature,
            train_target=train_target,
            val_feature=val_feature,
            val_target=val_target,
            batch_size=TRAINING_CONFIG['batch_size'],
            val_batch_size=TRAINING_CONFIG['val_batch_size']
        )
        print ("data processing success ...")

        # Initialize model
        model = HybridNetwork(
            input_size=train_feature.shape[1],
            output_size=1,
            num_layers=TRAINING_CONFIG['num_layers'],
            hidden_size=TRAINING_CONFIG['hidden_size'],
            dropout_rate=TRAINING_CONFIG['dropout_rate'],
            leaky_relu_slope=TRAINING_CONFIG['leaky_relu_slope'],
            activation=TRAINING_CONFIG['activation']
        )
        print ("build model success ...")

        # Initialize model manager
        model_manager = ModelManager(base_dir)

        # Initialize trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=TRAINING_CONFIG,
            model_manager=model_manager
        )

        # Train model
        train_losses, val_losses = trainer.train(f"{model_name}.pth")

        # Evaluate model
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model(
            model=trainer.model,
            data_loader=val_loader,
            scale_factor=SCALE
        )
        evaluator.print_metrics(metrics)

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()