import numpy as np
from helpers import load_csv_data, create_csv_submission
import matplotlib.pyplot as plt
import seaborn as sns

# x_train_og, x_test_og, y_train_og, train_ids_og, test_ids_og = load_csv_data("dataset/")

def preprocess_data(x_train, x_test, fill_nan_values=True, strategy='mean', apply_correlation=True, normalize=True):
    """
    Preprocess the data by filling NaN values, removing correlated features, and normalizing.

    Parameters
    ----------
    x_train : np.array
        Training features
    x_test : np.array
        Test features
    fill_nan_values : bool
        Whether to fill NaN values
    strategy : str
        Strategy to fill NaN values: 'mean', 'median', 'zero', 'drop'
    apply_correlation : bool
        Whether to remove highly correlated features
    normalize : bool
        Whether to normalize features (z-score normalization)

    Returns
    -------
    x_train_processed, x_test_processed : Arrays with processed features
    """
    x_train_processed = x_train.copy()
    x_test_processed = x_test.copy()
    
    if fill_nan_values:

        if strategy == 'mean':
            feature_means = np.nanmean(x_train, axis=0) #nanmean ignores NaN values
            for col in range(x_train.shape[1]):
                # Traiter train
                nan_mask = np.isnan(x_train_processed[:, col]) 
                x_train_processed[nan_mask, col] = feature_means[col]
                
                # Traiter test 
                nan_mask = np.isnan(x_test_processed[:, col])
                x_test_processed[nan_mask, col] = feature_means[col]

        elif strategy == 'median':
            feature_medians = np.nanmedian(x_train, axis=0)
            for col in range(x_train.shape[1]):
                # Traiter train
                nan_mask = np.isnan(x_train_processed[:, col])
                x_train_processed[nan_mask, col] = feature_medians[col]
                
                # Traiter test 
                nan_mask = np.isnan(x_test_processed[:, col])
                x_test_processed[nan_mask, col] = feature_medians[col]

        elif strategy == 'zero':
            x_train_processed[np.isnan(x_train_processed)] = 0
            x_test_processed[np.isnan(x_test_processed)] = 0

        elif strategy == 'drop':
            valid_features = ~np.isnan(x_train_processed).any(axis=0) # ~ inverse les boolean
            x_train_processed = x_train_processed[:, valid_features]
            x_test_processed = x_test_processed[:, valid_features]

        print(f"Filled NaN values using strategy: {strategy}")
        print(f"Remaining NaN in training set: {np.sum(np.isnan(x_train_processed))}")
        print(f"Remaining NaN in test set: {np.sum(np.isnan(x_test_processed))}")


    if apply_correlation: 
        correlation_matrix = np.corrcoef(x_train_processed, rowvar=False)
        
        # Visualize correlation matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
        
        threshold_high = 0.75

        print(f"Number of features before removing highly correlated features: {x_train_processed.shape[1]}")

        high_corr_pairs = [
            (i, j)
            for i in range(correlation_matrix.shape[0])
            for j in range(i + 1, correlation_matrix.shape[1])
            if abs(correlation_matrix[i, j]) > threshold_high
        ]
        
        features_to_drop = set()
        for i, j in high_corr_pairs:
            features_to_drop.add(j)  # Arbitrarily drop the second feature in the pair

        # Remove the selected features by index
        features_to_keep = [k for k in range(x_train_processed.shape[1]) if k not in features_to_drop]
        x_train_processed = x_train_processed[:, features_to_keep]
        x_test_processed = x_test_processed[:, features_to_keep]
                
        print(f"Number of features after removing highly correlated features: {x_train_processed.shape[1]}")
        print("Highly correlated features:")
        for i, j in high_corr_pairs:
            print(f"Feature {i} and Feature {j}: {correlation_matrix[i, j]}")
        plt.show()
    
    if normalize:
        # Use training data statistics to normalize both train and test
        feature_means = np.mean(x_train_processed, axis=0)
        feature_stds = np.std(x_train_processed, axis=0)
        
        # Avoid division by zero for constant features
        feature_stds = np.where(feature_stds == 0, 1, feature_stds)
        
        x_train_processed = (x_train_processed - feature_means) / feature_stds
        x_test_processed = (x_test_processed - feature_means) / feature_stds
        
        print(f"Features normalized. Train mean: {np.mean(x_train_processed):.6f}, Train std: {np.std(x_train_processed):.6f}")
        
    return x_train_processed, x_test_processed

def one_hot_encode(column):
    """
    Convert a categorical column to one-hot encoded representation.
    
    This function creates a binary matrix representation of categorical data,
    where each unique category is represented by a separate binary column.
    Uses k-1 encoding (drops the last category) to avoid multicollinearity.
    
    Parameters
    ----------
    column : np.array
        1D array containing categorical values to be one-hot encoded.
        Can contain any hashable data type (strings, integers, etc.).
    
    Returns
    -------
    np.array
        2D binary matrix of shape (n_samples, n_categories-1) where:
        - Each row corresponds to a sample from the input column
        - Each column corresponds to a unique category (except the last one)
        - Values are 1 if the sample belongs to that category, 0 otherwise
        - The last category is implicitly represented when all columns are 0
    
    Examples
    --------
    >>> categories = np.array(['A', 'B', 'C', 'A', 'B']).reshape((5, 1))
    >>> encoded = one_hot_encode(categories)
    >>> print(encoded)
    [[1. 0.]
     [0. 1.]
     [0. 0.]
     [1. 0.]
     [0. 1.]]
    """
    n = column.shape[0]
    unique_elements = np.unique(column)

    ohe_matrix = np.zeros(shape = (n, len(unique_elements) - 1))
    for i in range(n):
        idx = np.where(unique_elements == column[i])[0][0]
        if (idx != len(unique_elements) - 1):
            ohe_matrix[i, idx] = 1

    return ohe_matrix

def undersampling(seed = 42):
    """
    TODO: Implement undersampling functionality.
    """
    pass


def oversampling(seed = 42):
    """
    TODO: Implement oversampling functionality.
    """
    pass




# if __name__ == "__main__":
#     x_train, x_test = preprocess_data(x_train_og, x_test_og, fill_nan_values=True, strategy='mean', apply_correlation=True, normalize=True)
