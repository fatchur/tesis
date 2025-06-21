import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List
from config.config import SCALE, VERSION, SELECT_FEATURE, UPPER_THD, LOWER_THD, RANGES


class DataProcessor:
    """Handles data preparation and loading with range-based sampling"""
    @staticmethod
    def load_data(base_dir: str, target_col_name: str):
        """Load and preprocess training and validation data"""

        # Load and process training data
        train = pd.read_csv(f"{base_dir}/data/v{VERSION}/v{VERSION}_train.csv")
        train = train.dropna()

        # Load and process validation data
        val = pd.read_csv(f"{base_dir}/data/v{VERSION}/v{VERSION}_val.csv")
        val = val.dropna()

        # Load and process validation data
        test = pd.read_csv(f"{base_dir}/data/v{VERSION}/v{VERSION}_test.csv")
        test = test.dropna()

        f_columns = [col for col in train.columns if col.startswith('f')]
        feature_lst = []
        for col in f_columns:
            if pd.api.types.is_numeric_dtype(train[col]):
                feature_lst.append(col)

        train_feature = train[feature_lst]
        val_feature = val[feature_lst]
        test_feature = test[feature_lst]

        return train_feature, train[target_col_name], val_feature, val[target_col_name], test_feature, test[target_col_name]

    @staticmethod
    def prepare_data(
        train_feature: pd.DataFrame,
        train_target: pd.Series,
        val_feature: pd.DataFrame,
        val_target: pd.Series,
        test_feature: pd.DataFrame,
        test_target: pd.Series,
        batch_size: int = 64,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare train, validation, and test DataLoaders.
        
        Parameters:
        -----------
        train_feature : pd.DataFrame
            Training features
        train_target : pd.Series
            Training targets
        val_feature : pd.DataFrame
            Validation features
        val_target : pd.Series
            Validation targets
        test_feature : pd.DataFrame
            Test features
        test_target : pd.Series
            Test targets
        batch_size : int
            Batch size for training
        val_batch_size : Optional[int]
            Batch size for validation. If None, uses full dataset
        test_batch_size : Optional[int]
            Batch size for testing. If None, uses full dataset
        """
        
        X_train = torch.tensor(train_feature.values, dtype=torch.float32)
        y_train_labels = torch.tensor(train_target.values, dtype=torch.long)
        y_train = F.one_hot(y_train_labels, num_classes=len(RANGES)).float()
        
        train_dataset = TensorDataset(X_train, y_train)
        
        # Create validation dataset
        X_val = torch.tensor(val_feature.values, dtype=torch.float32)
        y_val_labels = torch.tensor(val_target.values, dtype=torch.long)
        y_val = F.one_hot(y_val_labels, num_classes=len(RANGES)).float()

        val_dataset = TensorDataset(X_val, y_val)

        # test dataset 
        X_test = torch.tensor(test_feature.values, dtype=torch.float32)
        y_test_labels = torch.tensor(test_target.values, dtype=torch.long)
        y_test = F.one_hot(y_test_labels, num_classes=len(RANGES)).float()

        test_dataset = TensorDataset(X_test, y_test)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True  # Drop last incomplete batch
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size or len(val_dataset),
            shuffle=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=test_batch_size or len(test_dataset),
            shuffle=False
        )

        return train_loader, val_loader, test_loader