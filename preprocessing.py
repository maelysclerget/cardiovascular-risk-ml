import numpy as np
from helpers import load_csv_data, create_csv_submission
import matplotlib.pyplot as plt
import seaborn as sns

# x_train_og, x_test_og, y_train_og, train_ids_og, test_ids_og = load_csv_data("dataset/")


def preprocess_data(
    x_train,
    x_test,
    fill_nan_values=True,
    strategy="mean",
    apply_correlation=True,
    normalize=True,
):
    """
    Preprocess the data by filling NaN values, removing correlated features, and normalizing.

    This function applies a sequence of preprocessing steps to the input data, including:
    - Filling NaN values using a specified strategy
    - Removing highly correlated features
    - Normalizing features (z-score normalization)

    Parameters
    ----------
    x_train : (np.array) (n_samples, n_features)
        Training features
    x_test : (np.array) (n_samples, n_features)
        Test features
    fill_nan_values : (bool)
        Whether to fill NaN values
    strategy : (str)
        Strategy to fill NaN values: 'mean', 'median', 'zero', 'drop'
    apply_correlation : (bool)
        Whether to remove highly correlated features
    normalize : (bool)
        Whether to normalize features (z-score normalization)

    Returns
    -------
    (tuple) (x_train_processed, x_test_processed) where:
        - x_train_processed: (np.array) Preprocessed training features
        - x_test_processed: (np.array) Preprocessed test features
    """
    x_train_processed = x_train.copy()
    x_test_processed = x_test.copy()

    if fill_nan_values:

        if strategy == "mean":
            feature_means = np.nanmean(x_train, axis=0)  # nanmean ignores NaN values
            for col in range(x_train.shape[1]):
                # Traiter train
                nan_mask = np.isnan(x_train_processed[:, col])
                x_train_processed[nan_mask, col] = feature_means[col]

                # Traiter test
                nan_mask = np.isnan(x_test_processed[:, col])
                x_test_processed[nan_mask, col] = feature_means[col]

        elif strategy == "median":
            feature_medians = np.nanmedian(x_train, axis=0)
            for col in range(x_train.shape[1]):
                # Traiter train
                nan_mask = np.isnan(x_train_processed[:, col])
                x_train_processed[nan_mask, col] = feature_medians[col]

                # Traiter test
                nan_mask = np.isnan(x_test_processed[:, col])
                x_test_processed[nan_mask, col] = feature_medians[col]

        elif strategy == "zero":
            x_train_processed[np.isnan(x_train_processed)] = 0
            x_test_processed[np.isnan(x_test_processed)] = 0

        elif strategy == "drop":
            valid_features = ~np.isnan(x_train_processed).any(
                axis=0
            )  # ~ inverse les boolean
            x_train_processed = x_train_processed[:, valid_features]
            x_test_processed = x_test_processed[:, valid_features]

        print(f"Filled NaN values using strategy: {strategy}")
        print(f"Remaining NaN in training set: {np.sum(np.isnan(x_train_processed))}")
        print(f"Remaining NaN in test set: {np.sum(np.isnan(x_test_processed))}")

    if apply_correlation:
        correlation_matrix = np.corrcoef(x_train_processed, rowvar=False)

        # Visualize correlation matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False)

        threshold_high = 0.75

        print(
            f"Number of features before removing highly correlated features: {x_train_processed.shape[1]}"
        )

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
        features_to_keep = [
            k for k in range(x_train_processed.shape[1]) if k not in features_to_drop
        ]
        x_train_processed = x_train_processed[:, features_to_keep]
        x_test_processed = x_test_processed[:, features_to_keep]

        print(
            f"Number of features after removing highly correlated features: {x_train_processed.shape[1]}"
        )
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

        print(
            f"Features normalized. Train mean: {np.mean(x_train_processed):.6f}, Train std: {np.std(x_train_processed):.6f}"
        )

    return x_train_processed, x_test_processed


def normalize(x_train, x_test, idx):
    """
    Normalize selected features in the training and test sets using z-score normalization.

    This function normalizes only the features specified by idx, using the mean and std of the training set.
    Constant features (std=0) are left unchanged.

    Parameters
    ----------
    x_train : (np.array) (n_samples, n_features)
        Training features
    x_test : (np.array) (n_samples, n_features)
        Test features
    idx : (list or np.array)
        Indices of features to normalize

    Returns
    -------
    (tuple) (x_train_processed, x_test_processed) where:
        - x_train_processed: (np.array) Normalized training features (selected columns only)
        - x_test_processed: (np.array) Normalized test features (selected columns only)

    Examples
    --------
    >>> import numpy as np
    >>> x_train = np.array([[1., 2.], [3., 4.], [5., 6.]])
    >>> x_test = np.array([[7., 8.], [9., 10.]])
    >>> idx = [0, 1]
    >>> x_train_norm, x_test_norm = normalize(x_train, x_test, idx)
    >>> np.round(x_train_norm, 2)
    array([[-1.22, -1.22],
           [ 0.  ,  0.  ],
           [ 1.22,  1.22]])
    >>> np.round(x_test_norm, 2)
    array([[2.45, 2.45],
           [3.67, 3.67]])
    """
    x_train_basis, x_test_basis = x_train[:, idx], x_test[:, idx]

    feature_means = np.mean(x_train_basis, axis=0)
    feature_stds = np.std(x_train_basis, axis=0)

    # Avoid division by 0 when the standar deviation is 0
    feature_stds = np.where(feature_stds == 0, 1, feature_stds)

    x_train_processed = (x_train_basis - feature_means) / feature_stds
    x_test_processed = (x_test_basis - feature_means) / feature_stds

    return x_train_processed, x_test_processed


def one_hot_encode(column, categories=None):
    """
    Convert a categorical column to one-hot encoded representation using k-1 encoding.

    This function creates a binary matrix representation of categorical data,
    where each unique category is represented by a separate binary column.
    Uses k-1 encoding (drops the last category) to avoid multicollinearity.

    Parameters
    ----------
    column : (np.array) (n_samples, 1) 1D array containing categorical values to be encoded
    categories : (np.array), optional
        Pre-defined categories to use for encoding. If None, will be derived from column.
        When provided, ensures consistent encoding across different data splits.

    Returns
    -------
    (np.array) (n_samples, n_categories - 1) 2D binary matrix where:
        - Each row corresponds to a sample from the input column
        - Each column corresponds to a unique category (except the last one)
        - Values are 1 if the sample belongs to that category, 0 otherwise
        - The last category is implicitly represented when all columns are 0
    """
    n = column.shape[0]
    if categories is None:
        categories = np.unique(column)

    # Ensure all values in column are present in categories
    if not np.all(np.isin(column, categories)):
        raise ValueError("Column contains values not present in provided categories")

    ohe_matrix = np.zeros(shape=(n, len(categories) - 1))
    for i in range(n):
        idx = np.where(categories == column[i])[0][0]
        if idx != len(categories) - 1:
            ohe_matrix[i, idx] = 1

    return ohe_matrix


def OHE_data(x_train, x_test, idx):
    """
    One-hot encode selected categorical features in train and test sets and concatenate them.

    This function applies one-hot encoding (using k-1 encoding) to each feature specified by idx,
    and concatenates the resulting binary columns for both train and test sets. It ensures
    consistent encoding between train and test by using combined unique categories.

    Parameters
    ----------
    x_train : (np.array) (n_samples_train, n_features)
        Training data
    x_test : (np.array) (n_samples_test, n_features)
        Test data
    idx : (list or np.array)
        Indices of features to one-hot encode

    Returns
    -------
    (tuple) (OHE_features_train, OHE_features_test) where:
        - OHE_features_train: (np.array) One-hot encoded features for train set
        - OHE_features_test: (np.array) One-hot encoded features for test set
    """
    x_train_basis, x_test_basis = x_train[:, idx], x_test[:, idx]
    d = x_train_basis.shape[1]
    OHE_features_train = []
    OHE_features_test = []

    for feature in range(d):
        # Get combined unique categories from both train and test
        train_feature = x_train_basis[:, feature]
        test_feature = x_test_basis[:, feature]
        all_categories = np.unique(np.concatenate([train_feature, test_feature]))

        # Encode both train and test using the same categories
        train_encoded = one_hot_encode(train_feature, categories=all_categories)
        test_encoded = one_hot_encode(test_feature, categories=all_categories)

        OHE_features_train.append(train_encoded)
        OHE_features_test.append(test_encoded)

    return np.column_stack(OHE_features_train), np.column_stack(OHE_features_test)


def undersampling(x, y, factor=1, seed=42):
    """
    Balance dataset by randomly removing samples from the majority class.

    This function reduces class imbalance by undersampling the majority class
    to match the minority class size, creating a balanced dataset.

    Parameters
    ----------
    x : (np.array) (n_samples, n_features) Input feature matrix
    y : (np.array) (n_samples,) Target labels corresponding to training data
    factor : (int) Reduction factor for undersampling the majority class
    seed : (int) Random seed for reproducible sampling

    Returns
    -------
    (tuple) (train_reduced, test_reduced) where:
        - x_reduced: (np.array) Input data with majority class samples removed
        - y_reduced: (np.array) Corresponding labels with majority class samples removed

    Examples
    --------
    >>> np.random.seed(42)
    >>> X = np.random.rand(100, 3)
    >>> y = np.array([0]*20 + [1]*80)  # Imbalanced: 20 class 0, 80 class 1
    >>> X_bal, y_bal = undersampling(X, y, seed=42)
    >>> np.unique(y_bal, return_counts=True)
    (array([0, 1]), array([20, 20]))
    """
    values, counts = np.unique(y, return_counts=True)

    lower = counts[np.argmin(counts)]
    higher = counts[np.argmax(counts)]
    high_val = values[np.argmax(counts)]

    high_idx = np.where(y == high_val)[0]

    fold = np.floor(higher / lower)
    n_drop = np.floor((fold - 1) * higher / (factor * fold))

    rng = np.random.default_rng(seed)
    drop_columns_idx = rng.choice(high_idx, size=int(n_drop), replace=False)

    x_reduced = np.delete(x, drop_columns_idx, axis=0)
    y_reduced = np.delete(y, drop_columns_idx, axis=0)

    return x_reduced, y_reduced


def oversampling(x, y, factor=1):
    """
    Balance dataset by duplicating samples from the minority class.

    This function reduces class imbalance by oversampling the minority class
    to match the majority class size, creating a balanced dataset.

    Parameters
    ----------
    x : (np.array) (n_samples, n_features) Input feature matrix
    y : (np.array) (n_samples,) Target labels corresponding to training data
    factor : (int) Multiplicative factor for oversampling the minority class

    Returns
    -------
    (tuple) (train_augmented, test_augmented) where:
        - x_augmented: (np.array) Input data with minority class samples duplicated
        - y_augmented: (np.array) Corresponding labels with minority class samples duplicated

    Examples
    --------
    >>> np.random.seed(42)
    >>> X = np.random.rand(30, 2)
    >>> y = np.array([0]*10 + [1]*20)  # Imbalanced: 10 class 0, 20 class 1
    >>> X_bal, y_bal = oversampling(X, y)
    >>> np.unique(y_bal, return_counts=True)
    (array([0, 1]), array([20, 20]))
    """
    values, counts = np.unique(y, return_counts=True)

    lower = counts[np.argmin(counts)]
    low_val = values[np.argmin(counts)]
    higher = counts[np.argmax(counts)]

    low_idx = np.where(y == low_val)[0]

    fold = np.floor(higher * factor / lower)

    x_duplicated_points = np.repeat(x[low_idx], int(fold - 1), axis=0)
    y_duplicated_points = np.repeat(y[low_idx], int(fold - 1), axis=0)

    x_augmented = np.concatenate((x, x_duplicated_points), axis=0)
    y_augmented = np.concatenate((y, y_duplicated_points), axis=0)

    return x_augmented, y_augmented


def preprocessing_pipeline(
    x_train,
    x_test,
    OHE_idx,
    normalization_idx,
    apply_correlation=True,
    corr_threshold=0.75,
):
    """
    Apply a preprocessing pipeline with normalization, one-hot encoding, and optional correlation filtering.

    This function performs the following steps:
    - One-hot encodes the features specified by OHE_idx
    - Normalizes the features specified by normalization_idx using z-score normalization
    - Concatenates the normalized and one-hot encoded features
    - Optionally removes highly correlated normalized features above the given threshold

    Parameters
    ----------
    x_train : (np.array) (n_samples, n_features)
        Training features
    x_test : (np.array) (n_samples, n_features)
        Test features
    OHE_idx : (list or np.array)
        Indices of features to one-hot encode
    normalization_idx : (list or np.array)
        Indices of features to normalize
    apply_correlation : (bool), optional
        Whether to remove highly correlated normalized features (default: True)
    corr_threshold : (float), optional
        Threshold above which features are considered highly correlated and removed (default: 0.75)

    Returns
    -------
    (tuple) (x_train_processed, x_test_processed, normalized_cols_idx) where:
        - x_train_processed: (np.array) Preprocessed training features
        - x_test_processed: (np.array) Preprocessed test features
        - normalized_cols_idx: (np.array) Indices of the normalized columns after correlation filtering
    """
    x_train_OHE, x_test_OHE = OHE_data(x_train, x_test, OHE_idx)
    x_train_norm, x_test_norm = normalize(x_train, x_test, normalization_idx)

    x_train_processed = np.concatenate([x_train_norm, x_train_OHE], axis=1)
    x_test_processed = np.concatenate([x_test_norm, x_test_OHE], axis=1)

    print(
        f"Number of normalized features: {len(normalization_idx)}, Number of OHE features: {len(OHE_idx)}",
        "\n",
    )
    print(
        f"The first {len(normalization_idx)} columns contain the normalized indices. The rest of the columns are OHE.",
        "\n",
    )

    normalized_cols_idx = np.arange(len(normalization_idx))
    normalized_cols = x_train_processed[:, normalized_cols_idx]

    if apply_correlation:
        correlation_matrix = np.corrcoef(normalized_cols, rowvar=False)

        # Visualize correlation matrix
        # plt.figure(figsize=(12, 10))
        # sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)

        print(
            f"Number of features before removing highly correlated features: {x_train_processed.shape[1]}"
        )

        high_corr_pairs = [
            (i, j)
            for i in range(correlation_matrix.shape[0])
            for j in range(i + 1, correlation_matrix.shape[1])
            if abs(correlation_matrix[i, j]) > corr_threshold
        ]

        features_to_drop = set()
        for i, j in high_corr_pairs:
            features_to_drop.add(j)  # Arbitrarily drop the second feature in the pair

        x_train_processed = np.delete(
            x_train_processed, np.array(list(features_to_drop)), axis=1
        )
        x_test_processed = np.delete(
            x_test_processed, np.array(list(features_to_drop)), axis=1
        )

        normalized_cols_idx = np.arange(len(normalization_idx) - len(features_to_drop))

        print(
            f"Number of features after removing highly correlated features: {x_train_processed.shape[1]}"
        )
        print("Highly correlated features:")
        for i, j in high_corr_pairs:
            print(f"Feature {i} and Feature {j}: {correlation_matrix[i, j]}")
        plt.show()

    print(
        f"Number of normalized features: {len(normalized_cols_idx)}, Number of OHE features: {len(OHE_idx)}",
        "\n",
    )
    print(
        f"The first {len(normalized_cols_idx)} columns contain the normalized indices. The rest of the columns are OHE.",
        "\n",
    )

    return x_train_processed, x_test_processed, normalized_cols_idx


# if __name__ == "__main__":
# x_train, x_test = preprocess_data(x_train_og, x_test_og, fill_nan_values=True, strategy='mean', apply_correlation=True, normalize=True)
# x_train, x_test, normalized_cols_idx = preprocessing_pipeline(x_train_og, x_test_og, OHE_idx = [0, 1, 2], normalization_idx = [3, 4])
