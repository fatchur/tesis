import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch
from typing import Tuple, Optional
from config.config import SCALE, VERSION, SELECT_FEATURE, UPPER_THD, LOWER_THD

class DataProcessor:
    """Handles data preparation and loading"""
    @staticmethod
    def load_data():
        """Load and preprocess training and validation data"""
        # Load feature importance data
        important_feature = pd.read_csv(f"data/v{VERSION}/v{VERSION}_correlations.csv")
        feature_lst = important_feature['Unnamed: 0'].tolist()

        # Load and process training data
        train = pd.read_csv(f"data/v{VERSION}/v{VERSION}_train.csv")
        train = train.dropna()
        train['gd'] = train['gd'] / SCALE
        train = train[(train['gd'] <= UPPER_THD) & (train['gd'] >= LOWER_THD)]

        # Load and process validation data
        val = pd.read_csv(f"data/v{VERSION}/v{VERSION}_val.csv")
        val = val.dropna()
        val['gd'] = val['gd'] / SCALE
        val = val[(val['gd'] <= UPPER_THD) & (val['gd'] >= LOWER_THD)]

        # Select features
        drop_columns = ['gd', 'id', 'Unnamed: 0', 'id_pasien']
        if SELECT_FEATURE:
            train_feature = train[feature_lst]
            val_feature = val[feature_lst]
        else:
            train_feature = train.drop(drop_columns, axis=1)
            val_feature = val.drop(drop_columns, axis=1)

        return train_feature, train['gd'], val_feature, val['gd']

    @staticmethod
    def prepare_data(
        train_feature: pd.DataFrame,
        train_target: pd.Series,
        val_feature: pd.DataFrame,
        val_target: pd.Series,
        batch_size: int = 64,
        val_batch_size: Optional[int] = None
    ) -> Tuple[DataLoader, DataLoader]:
        """Prepare train and validation DataLoaders"""
        # Convert to tensors
        X_train = torch.tensor(train_feature.values, dtype=torch.float32, requires_grad=True)
        y_train = torch.tensor(train_target.values, dtype=torch.float32, requires_grad=True).view(-1, 1)
        X_val = torch.tensor(val_feature.values, dtype=torch.float32, requires_grad=True)
        y_val = torch.tensor(val_target.values, dtype=torch.float32, requires_grad=True).view(-1, 1)

        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size or len(val_dataset),
            shuffle=False
        )

        return train_loader, val_loader