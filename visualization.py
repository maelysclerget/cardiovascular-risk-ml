import numpy as np 
import matplotlib.pyplot as plt
import os
from helpers import load_csv_data

def load_data(data_path="dataset/", sub_sample=False):
    """
    Load the dataset using the helper function.
    
    Parameters
    ----------
    data_path : str
        Path to the dataset folder (default: "dataset/")
    sub_sample : bool
        Whether to subsample the data (default: False)
    
    Returns
    -------
    x_train, x_test, y_train, train_ids, test_ids : numpy arrays
    """
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path, sub_sample)
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Training IDs shape: {train_ids.shape}")
    print(f"Test IDs shape: {test_ids.shape}")
    
    return x_train, x_test, y_train, train_ids, test_ids



def visualize_data_overview(x_train, y_train):
    """
    Create overview visualizations of the dataset.
    
    Parameters
    ----------
    x_train : np.array
        Training features
    y_train : np.array  
        Training labels
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Label distribution
    unique, counts = np.unique(y_train, return_counts=True)
    axes[0, 0].bar(unique, counts, color=['red', 'blue'])
    axes[0, 0].set_title('Label Distribution')
    axes[0, 0].set_xlabel('Label')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_xticks(unique)
    
    # Add percentage labels on bars
    total = len(y_train)
    for i, (label, count) in enumerate(zip(unique, counts)):
        percentage = (count / total) * 100
        axes[0, 0].text(label, count + 50, f'{percentage:.1f}%', 
                       ha='center', va='bottom')
    
    # 2. Feature statistics
    feature_means = np.mean(x_train, axis=0)
    feature_stds = np.std(x_train, axis=0)
    
    axes[0, 1].hist(feature_means, bins=50, alpha=0.7, color='green')
    axes[0, 1].set_title('Distribution of Feature Means')
    axes[0, 1].set_xlabel('Mean Value')
    axes[0, 1].set_ylabel('Number of Features')
    
    # 3. Feature standard deviations
    axes[1, 0].hist(feature_stds, bins=50, alpha=0.7, color='orange')
    axes[1, 0].set_title('Distribution of Feature Standard Deviations')
    axes[1, 0].set_xlabel('Standard Deviation')
    axes[1, 0].set_ylabel('Number of Features')
    
    # 4. Missing values heatmap (sample)
    # Check for missing values (NaN or very large values that might indicate missing)
    missing_mask = np.isnan(x_train) | (np.abs(x_train) > 1e6)
    missing_per_feature = np.sum(missing_mask, axis=0)
    
    axes[1, 1].plot(missing_per_feature)
    axes[1, 1].set_title('Missing Values per Feature')
    axes[1, 1].set_xlabel('Feature Index')
    axes[1, 1].set_ylabel('Number of Missing Values')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n=== DATA OVERVIEW ===")
    print(f"Number of samples: {x_train.shape[0]}")
    print(f"Number of features: {x_train.shape[1]}")
    print(f"Label distribution: {dict(zip(unique, counts))}")
    print(f"Features with missing values: {np.sum(missing_per_feature > 0)}")



def basic_data_stats(x_train, y_train):
    """
    Print basic statistics about the data.
    
    Parameters
    ----------
    x_train : np.array
        Training features
    y_train : np.array
        Training labels
    """
    print("\n=== BASIC STATISTICS ===")
    print(f"Dataset shape: {x_train.shape}")
    print(f"Feature value range: [{np.min(x_train):.2f}, {np.max(x_train):.2f}]")
    print(f"Feature mean: {np.mean(x_train):.2f}")
    print(f"Feature std: {np.std(x_train):.2f}")
    
    # Check for potential issues
    print(f"\nFeatures with zero variance: {np.sum(np.var(x_train, axis=0) == 0)}")
    print(f"Features with very high values (>1000): {np.sum(np.max(x_train, axis=0) > 1000)}")
    print(f"Features with negative values: {np.sum(np.min(x_train, axis=0) < 0)}")
    
    # Label statistics  
    unique_labels, counts = np.unique(y_train, return_counts=True)
    print(f"\nLabel classes: {unique_labels}")
    print(f"Label counts: {counts}")
    print(f"Class balance: {counts[1]/counts[0]:.2f} (pos/neg ratio)")


if __name__ == "__main__":
    x_train, x_test, y_train, train_ids, test_ids = load_data()
    #basic_data_stats(x_train, y_train)
    #visualize_data_overview(x_train, y_train)