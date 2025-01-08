import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import torch
from typing import Tuple, Optional, List
from config.config import SCALE, VERSION, SELECT_FEATURE, UPPER_THD, LOWER_THD, RANGES


def normalize_target(value: float, ranges: List[Tuple[float, float]], scale: float) -> List[float]:
    """
    Normalize glucose value into target vector where:
    - Only one position has non-zero value (the range it falls into)
    - The value is normalized within that range
    
    Args:
        value: Glucose value (already normalized by SCALE)
        ranges: List of (start, end) ranges
        scale: SCALE value for denormalization if needed
    
    Returns:
        List of floats representing target vector
    """
    result = [0.0] * len(ranges)
    
    for i, (start, end) in enumerate(ranges):
        if start <= value < end:
            if end == float('inf'):
                # For last range (>300/SCALE)
                # Normalize between start and 1.0
                normalized = (value - start)/(1.0 - start)
            else:
                # For other ranges
                # Normalize between 0 and 1 within range
                normalized = (value - start)/(end - start)
            
            result[i] = normalized
            break
    
    return result

class DataProcessor:
    """Handles data preparation and loading with range-based sampling"""
    @staticmethod
    def load_data(base_dir: str):
        """Load and preprocess training and validation data"""
        # Load feature importance data
        important_feature = pd.read_csv(f"{base_dir}/data/v{VERSION}/v{VERSION}_correlations.csv")
        feature_lst = important_feature['Unnamed: 0'].tolist()

        # Load and process training data
        train = pd.read_csv(f"{base_dir}/data/v{VERSION}/v{VERSION}_train.csv")
        train = train.dropna()
        train['gd'] = train['gd'] / SCALE
        train = train[(train['gd'] <= UPPER_THD) & (train['gd'] >= LOWER_THD)]

        # Load and process validation data
        val = pd.read_csv(f"{base_dir}/data/v{VERSION}/v{VERSION}_val.csv")
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
    def create_range_datasets(
        features: pd.DataFrame,
        targets: pd.Series,
        samples_per_range: Optional[int] = None
    ) -> List[TensorDataset]:
        """
        Split data into ranges and create datasets
        
        Parameters:
        -----------
        features : pd.DataFrame
            Feature data
        targets : pd.Series
            Target values
        samples_per_range : Optional[int]
            Number of samples to take from each range. If None, uses all samples.
        
        Returns:
        --------
        List[TensorDataset]
            List of datasets for each range
        """
        # Define value ranges (after scaling)
        ranges = RANGES
        
        # Initialize range datasets
        range_datasets = []
        
        # Create masks for each range
        range_masks = [
            (targets >= start) & (targets < end)
            for start, end in ranges
        ]
        
        # Print original distribution
        print("\nOriginal distribution:")
        for i, ((start, end), mask) in enumerate(zip(ranges, range_masks), 1):
            count = mask.sum()
            start_val = start * SCALE
            end_val = end * SCALE if end != float('inf') else 'inf'
            print(f"Range {start_val:.0f} - {end_val}: {count} samples")
        
        for i, ((start, end), mask) in enumerate(zip(ranges, range_masks), 1):
            range_features = features[mask]
            range_targets = targets[mask]
            
            original_size = len(range_features)
            print(f"Range {i} original size: {original_size}")
            
            # Sample if specified
            if samples_per_range is not None:
                replace = len(range_features) < samples_per_range
                indices = np.random.choice(
                    len(range_features),
                    samples_per_range,
                    replace=replace
                )
                range_features = range_features.iloc[indices]
                range_targets = range_targets.iloc[indices]
                
                if replace:
                    print(f"Range {start * SCALE:.0f} - {end * SCALE if end != float('inf') else 'inf'}: Upsampled from {original_size} to {samples_per_range} samples (with replacement)")
                else:
                    print(f"Range {start * SCALE:.0f} - {end * SCALE if end != float('inf') else 'inf'}: Downsampled from {original_size} to {samples_per_range} samples")
            
            # Convert features to tensor
            X = torch.tensor(
                range_features.values,
                dtype=torch.float32,
                requires_grad=True
            )
            
            # Convert targets to normalized range vectors
            normalized_targets = np.array([
                normalize_target(target_val, RANGES, SCALE) 
                for target_val in range_targets.values
            ])
            
            # Convert to tensor
            y = torch.tensor(
                normalized_targets,
                dtype=torch.float32,
                requires_grad=True
            )
            
            range_datasets.append(TensorDataset(X, y))

        return range_datasets

    @staticmethod
    def prepare_data(
        train_feature: pd.DataFrame,
        train_target: pd.Series,
        val_feature: pd.DataFrame,
        val_target: pd.Series,
        batch_size: int = 64,
        val_batch_size: Optional[int] = None,
        samples_per_range: Optional[int] = None,
        use_weights: bool = True
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare train and validation DataLoaders with range-based sampling
        
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
        batch_size : int
            Batch size for training
        val_batch_size : Optional[int]
            Batch size for validation. If None, uses full dataset
        samples_per_range : Optional[int]
            Number of samples to take from each range. If None, uses proportional sampling
        """
        # Create range datasets for training data
        if samples_per_range is None:
            # Calculate samples per range (use minimum size of all ranges)
            ranges = RANGES
            range_sizes = [
                ((train_target >= start) & (train_target < end)).sum()
                for start, end in ranges
            ]
            samples_per_range = max(range_sizes)
        
        range_datasets = DataProcessor.create_range_datasets(
            train_feature,
            train_target,
            samples_per_range
        )
        
        # Combine all data from ranges into single tensors
        all_features = []
        all_targets = []
        for dataset in range_datasets:
            features, targets = dataset.tensors
            all_features.append(features)
            all_targets.append(targets)
        
        # Concatenate into single tensors
        all_features = torch.cat(all_features, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Create single dataset
        combined_train_dataset = TensorDataset(all_features, all_targets)
        
        # Create validation dataset
        X_val = torch.tensor(
            val_feature.values,
            dtype=torch.float32,
            requires_grad=True
        )

        normalized_val_targets = np.array([
            normalize_target(target_val, RANGES, SCALE) 
            for target_val in val_target.values])

        # y_val = torch.tensor(
        #     val_target.values,
        #     dtype=torch.float32,
        #     requires_grad=True
        # ).view(-1, 1)

        y_val = torch.tensor(
                    normalized_val_targets,
                    dtype=torch.float32,
                    requires_grad=True
                )

        val_dataset = TensorDataset(X_val, y_val)

        # Create dataloaders
        train_loader = DataLoader(
            combined_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True  # Drop last incomplete batch
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size or len(val_dataset),
            shuffle=False
        )

        return train_loader, val_loader