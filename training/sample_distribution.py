import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_processing import DataProcessor
from config.config import SCALE

def analyze_distribution():
    # Load original data
    data_processor = DataProcessor()
    train_feature, train_target, _, _ = data_processor.load_data()
    
    # Get resampled data using our range approach
    samples_per_range = 2000  # Or your desired number
    train_loader, _ = data_processor.prepare_data(
        train_feature=train_feature,
        train_target=train_target,
        val_feature=train_feature,  # Dummy val data
        val_target=train_target,    # Dummy val data
        samples_per_range=None
    )
    
    # Extract all targets from the dataloader
    resampled_targets = []
    for _, targets in train_loader:
        resampled_targets.extend(targets.squeeze().tolist())
    
    # Define ranges for analysis
    ranges = [
        (0, 100),
        (100, 200),
        (200, 300),
        (300, float('inf'))
    ]
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Histogram plots
    orig_data = train_target * SCALE
    sns.histplot(data=orig_data, bins=50, ax=ax1[0])
    ax1[0].set_title('Original Target Distribution (Histogram)')
    ax1[0].set_xlabel('Target Value')
    ax1[0].set_ylabel('Count')
    
    resampled_data = np.array(resampled_targets) * SCALE
    sns.histplot(data=resampled_data, bins=50, ax=ax1[1])
    ax1[1].set_title('Resampled Target Distribution (Histogram)')
    ax1[1].set_xlabel('Target Value')
    ax1[1].set_ylabel('Count')
    
    # Bar plots for range counts
    def count_in_ranges(data, ranges):
        counts = []
        labels = []
        for start, end in ranges:
            if end == float('inf'):
                mask = (data >= start)
                labels.append(f'≥{start}')
            else:
                mask = (data >= start) & (data < end)
                labels.append(f'{start}-{end}')
            counts.append(np.sum(mask))
        return counts, labels
    
    # Original data range counts
    orig_counts, labels = count_in_ranges(orig_data, ranges)
    ax2[0].bar(labels, orig_counts)
    ax2[0].set_title('Original Sample Distribution by Range')
    ax2[0].set_ylabel('Count')
    ax2[0].tick_params(axis='x', rotation=45)
    
    # Resampled data range counts
    resampled_counts, _ = count_in_ranges(resampled_data, ranges)
    ax2[1].bar(labels, resampled_counts)
    ax2[1].set_title('Resampled Distribution by Range')
    ax2[1].set_ylabel('Count')
    ax2[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('distribution_comparison.png')
    
    # Print statistics
    print("\nDistribution Statistics:")
    print("\nOriginal Data:")
    print(f"Total samples: {len(orig_data)}")
    print(f"Mean: {orig_data.mean():.2f}")
    print(f"Std: {orig_data.std():.2f}")
    print("\nSamples in each range:")
    for (start, end), count in zip(ranges, orig_counts):
        if end == float('inf'):
            print(f"Range ≥{start}: {count} samples")
        else:
            print(f"Range {start}-{end}: {count} samples")
    
    print("\nResampled Data:")
    print(f"Total samples: {len(resampled_data)}")
    print(f"Mean: {resampled_data.mean():.2f}")
    print(f"Std: {resampled_data.std():.2f}")
    print("\nSamples in each range:")
    for (start, end), count in zip(ranges, resampled_counts):
        if end == float('inf'):
            print(f"Range ≥{start}: {count} samples")
        else:
            print(f"Range {start}-{end}: {count} samples")

if __name__ == "__main__":
    analyze_distribution()