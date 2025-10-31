"""
Specialized data filling and processing module for continuous feature optimization.
This module implements advanced imputation techniques tailored for continuous features.
"""

import numpy as np
from typing import List, Dict, Optional
import os

# Load filled datasets once (global variables)
print("Loading filled datasets...")


# Function to read CSV with header
def read_csv_with_header(filepath: str):
    # Read data with numpy
    data = np.genfromtxt(filepath, delimiter=",", skip_header=1)
    # Read header
    with open(filepath, "r") as f:
        header = f.readline().strip().split(",")
    return data, header


# Load training data
print("Loading training data...")
train_data, train_header = read_csv_with_header(
    "dataset_features_removed/x_train_filled.csv"
)
train_ids_og = train_data[:, 0].astype(int)
x_train_og = train_data[:, 1:]  # Exclude Id column

# Load test data
print("Loading test data...")
test_data, test_header = read_csv_with_header(
    "dataset_features_removed/x_test_filled.csv"
)
test_ids_og = test_data[:, 0].astype(int)
x_test_og = test_data[:, 1:]  # Exclude Id column

# Load y_train data
print("Loading y_train data...")
y_data = np.genfromtxt(
    "dataset_features_removed/y_train.csv", delimiter=",", skip_header=1, usecols=1
)
y_train_og = y_data.copy()

print(f"Datasets loaded:")
print(f"Train: {x_train_og.shape}, Test: {x_test_og.shape}")


# %%
def simple_value_replacement(
    feature: np.ndarray, old_value: float, new_value: float
) -> None:
    """
    Replace a specific value with another value in a feature.
    MODIFIES THE INPUT ARRAY IN-PLACE (no copy made).

    Parameters
    ----------
    feature : np.ndarray
        The feature data (1D array) - MODIFIED IN-PLACE
    old_value : float
        Value to be replaced (e.g., 88 for "don't know")
    new_value : float
        Value to replace with (e.g., 0)

    Returns
    -------
    None
        Modifies data in-place

    Example
    -------
    >>> feature = np.array([1, 5, 88, 10, 88, 3])
    >>> simple_value_replacement(feature, old_value=88, new_value=0)
    >>> print(feature)  # [1, 5, 0, 10, 0, 3] - modified in-place!
    """
    # Handle NaN case
    if np.isnan(old_value):
        mask = np.isnan(feature)
    else:
        mask = feature == old_value

    feature[mask] = new_value


# %%
def redistribute_with_distribution(
    x_train_feature: np.ndarray,
    x_test_feature: np.ndarray,
    values_to_redistribute: List[float],
    valid_range: tuple,
    random_state: Optional[int] = None,
) -> tuple:
    """
    Redistribute special values (77, 99, BLANK) according to the distribution
    of valid values, preserving the original distribution.

    IMPORTANT: For x_test, we use the SAME distribution calculated from x_train
    (to avoid data leakage - we don't look at test data distribution).

    Parameters
    ----------
    x_train_feature : np.ndarray
        Training feature data (1D array)
    x_test_feature : np.ndarray
        Test feature data (1D array)
    values_to_redistribute : list of float
        Values to be redistributed (e.g., [77, 99, np.nan] for don't know, refused, blank)
    valid_range : tuple
        (min, max) of valid values (e.g., (1, 30) for days sick)
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    tuple : (x_train_filled, x_test_filled, distribution_info)
        - x_train_filled: Training data with redistributed values
        - x_test_filled: Test data with redistributed values (using train distribution)
        - distribution_info: Dict with distribution statistics

    Example
    -------
    >>> # PHYSHLTH: days feeling sick (1-30)
    >>> # 77 = don't know, 88 = none, 99 = refused, BLANK = missing
    >>>
    >>> x_train = np.array([1, 5, 10, 77, 99, 15, 20, np.nan, 5, 10])
    >>> x_test = np.array([3, 77, 12, 99, np.nan, 8])
    >>>
    >>> train_filled, test_filled, info = redistribute_with_distribution(
    ...     x_train, x_test,
    ...     values_to_redistribute=[77, 99, np.nan],
    ...     valid_range=(1, 30),
    ...     random_state=42
    ... )
    """
    if random_state is not None:
        np.random.seed(random_state)

    x_train_copy = x_train_feature.copy()
    x_test_copy = x_test_feature.copy()

    min_val, max_val = valid_range

    # Step 1: Identify valid values in training data
    valid_mask_train = np.ones(len(x_train_copy), dtype=bool)

    for special_val in values_to_redistribute:
        if np.isnan(special_val):
            valid_mask_train &= ~np.isnan(x_train_copy)
        else:
            valid_mask_train &= x_train_copy != special_val

    # Also filter by valid range
    valid_mask_train &= (x_train_copy >= min_val) & (x_train_copy <= max_val)

    # Get valid values from training data
    valid_train_values = x_train_copy[valid_mask_train]

    if len(valid_train_values) == 0:
        raise ValueError(
            "No valid values found in training data to compute distribution!"
        )

    # Step 2: Calculate distribution from training data
    unique_values, counts = np.unique(valid_train_values, return_counts=True)
    probabilities = counts / counts.sum()

    print(
        f"Distribution calculated from {len(valid_train_values)} valid training samples"
    )
    print(f"Valid values range: {unique_values.min():.0f} to {unique_values.max():.0f}")
    print(f"Number of unique valid values: {len(unique_values)}")

    # Step 3: Identify values to redistribute in TRAINING data
    redistribute_mask_train = np.zeros(len(x_train_copy), dtype=bool)
    for special_val in values_to_redistribute:
        if np.isnan(special_val):
            redistribute_mask_train |= np.isnan(x_train_copy)
        else:
            redistribute_mask_train |= x_train_copy == special_val

    num_to_replace_train = redistribute_mask_train.sum()
    print(f"Training: Redistributing {num_to_replace_train} values")

    # Step 4: Redistribute in TRAINING data using training distribution
    if num_to_replace_train > 0:
        replacements_train = np.random.choice(
            unique_values, size=num_to_replace_train, p=probabilities
        )
        x_train_copy[redistribute_mask_train] = replacements_train

    # Step 5: Identify values to redistribute in TEST data
    redistribute_mask_test = np.zeros(len(x_test_copy), dtype=bool)
    for special_val in values_to_redistribute:
        if np.isnan(special_val):
            redistribute_mask_test |= np.isnan(x_test_copy)
        else:
            redistribute_mask_test |= x_test_copy == special_val

    num_to_replace_test = redistribute_mask_test.sum()
    print(f"Test: Redistributing {num_to_replace_test} values")

    # Step 6: Redistribute in TEST data using SAME training distribution
    if num_to_replace_test > 0:
        replacements_test = np.random.choice(
            unique_values, size=num_to_replace_test, p=probabilities
        )
        x_test_copy[redistribute_mask_test] = replacements_test

    # Distribution info
    distribution_info = {
        "unique_values": unique_values,
        "probabilities": probabilities,
        "n_valid_train_samples": len(valid_train_values),
        "n_redistributed_train": num_to_replace_train,
        "n_redistributed_test": num_to_replace_test,
    }

    return x_train_copy, x_test_copy, distribution_info


# %%
def preprocess_discrete_feature(
    x_train: np.ndarray,
    x_test: np.ndarray,
    feature_name: str,
    column_names: List[str],
    simple_replacements: Optional[Dict[float, float]] = None,
    redistribute_config: Optional[Dict] = None,
    random_state: Optional[int] = None,
) -> Dict:
    """
    Complete preprocessing pipeline for a discrete feature.
    Combines simple replacements and distribution-based redistribution.
    MODIFIES THE INPUT ARRAYS IN-PLACE (no copy made inside).

    Parameters
    ----------
    x_train : np.ndarray
        Training data (2D array) - MODIFIED IN-PLACE
    x_test : np.ndarray
        Test data (2D array) - MODIFIED IN-PLACE
    feature_name : str
        Name of the feature to process (e.g., 'PHYSHLTH')
    column_names : list of str
        List of all column names in the dataset
    simple_replacements : dict, optional
        Dictionary of {old_value: new_value} for simple replacements
        Example: {88: 0} to replace "don't know" with 0
    redistribute_config : dict, optional
        Configuration for redistribution:
        {
            'values_to_redistribute': [77, 99, np.nan],
            'valid_range': (1, 30)
        }
    random_state : int, optional
        Random seed

    Returns
    -------
    dict : distribution_info
        Dictionary with redistribution statistics (or None if no redistribution)

    Example
    -------
    >>> # IMPORTANT: Copy your data BEFORE calling this function
    >>> x_train_processed = x_train.copy()
    >>> x_test_processed = x_test.copy()
    >>>
    >>> # Process PHYSHLTH feature
    >>> # Step 1: Replace 88 (don't know) with 0 (no days sick)
    >>> # Step 2: Redistribute 77, 99, BLANK according to distribution of 1-30
    >>>
    >>> info = preprocess_discrete_feature(
    ...     x_train_processed, x_test_processed,  # These will be modified
    ...     feature_name='PHYSHLTH',
    ...     column_names=all_column_names,
    ...     simple_replacements={88: 0},
    ...     redistribute_config={
    ...         'values_to_redistribute': [77, 99, np.nan],
    ...         'valid_range': (1, 30)
    ...     },
    ...     random_state=42
    ... )
    >>> # Now x_train_processed and x_test_processed have been modified
    """
    # Get feature index from column name
    if feature_name not in column_names:
        raise ValueError(f"Feature '{feature_name}' not found in column_names")

    feature_index = column_names.index(feature_name)

    print(f"\n=== Processing Feature '{feature_name}' (index {feature_index}) ===")

    # Extract feature columns (these are VIEWS, not copies)
    train_feature = x_train[:, feature_index]
    test_feature = x_test[:, feature_index]

    # Step 1: Simple replacements (modifies in-place)
    if simple_replacements:
        print("Step 1: Applying simple replacements...")
        for old_val, new_val in simple_replacements.items():
            # Handle special 'nan' string key
            if old_val == "nan" or (
                isinstance(old_val, str) and old_val.lower() == "nan"
            ):
                print(f"  Replacing NaN → {new_val}")
                simple_value_replacement(train_feature, np.nan, new_val)
                simple_value_replacement(test_feature, np.nan, new_val)
            else:
                print(f"  Replacing {old_val} → {new_val}")
                simple_value_replacement(train_feature, old_val, new_val)
                simple_value_replacement(test_feature, old_val, new_val)

    # Step 2: Redistribution
    if redistribute_config:
        print("Step 2: Redistributing values according to distribution...")
        train_feature_new, test_feature_new, dist_info = redistribute_with_distribution(
            train_feature,
            test_feature,
            values_to_redistribute=redistribute_config["values_to_redistribute"],
            valid_range=redistribute_config["valid_range"],
            random_state=random_state,
        )
        # Update arrays with redistributed values
        x_train[:, feature_index] = train_feature_new
        x_test[:, feature_index] = test_feature_new
    else:
        dist_info = None

    print(f"=== Feature '{feature_name}' Processing Complete ===\n")

    return dist_info


# %%
def create_working_copies():
    """
    Create working copies of the original dataset for preprocessing.
    Call this function to get fresh copies you can modify.

    Returns
    -------
    tuple : (x_train, x_test, y_train, train_ids, test_ids)
        Fresh copies of the original data

    Example
    -------
    >>> # Create fresh copies to work with
    >>> x_train, x_test, y_train, train_ids, test_ids = create_working_copies()
    >>>
    >>> # Process features - modifies x_train and x_test in-place
    >>> preprocess_discrete_feature(x_train, x_test, feature_index=28, ...)
    >>> preprocess_discrete_feature(x_train, x_test, feature_index=29, ...)
    """
    return (
        x_train_og.copy(),
        x_test_og.copy(),
        y_train_og.copy(),
        train_ids_og.copy(),
        test_ids_og.copy(),
    )


# %%
def process_multiple_features(
    feature_configs: Dict[str, Dict],
    column_names: List[str],
    random_state: Optional[int] = None,
) -> tuple:
    """
    Process multiple discrete features at once using the original dataset.

    Parameters
    ----------
    feature_configs : dict
        Dictionary mapping feature NAMES to their preprocessing config:
        {
            'PHYSHLTH': {
                'simple_replacements': {88: 0},
                'redistribute_config': {
                    'values_to_redistribute': [77, 99, np.nan],
                    'valid_range': (1, 30)
                }
            },
            'MENTHLTH': {
                'simple_replacements': {88: 0},
                'redistribute_config': {
                    'values_to_redistribute': [77, 99, np.nan],
                    'valid_range': (1, 30)
                }
            }
        }
    column_names : list of str
        List of all column names in the dataset
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    tuple : (x_train_processed, x_test_processed, y_train, train_ids, test_ids, info_dict)
        Processed datasets and dictionary of distribution info for each feature

    Example
    -------
    >>> configs = {
    ...     'PHYSHLTH': {'simple_replacements': {88: 0},
    ...                  'redistribute_config': {'values_to_redistribute': [77, 99, np.nan], 'valid_range': (1, 30)}},
    ...     'MENTHLTH': {'simple_replacements': {88: 0},
    ...                  'redistribute_config': {'values_to_redistribute': [77, 99, np.nan], 'valid_range': (1, 30)}}
    ... }
    >>> x_train, x_test, y_train, ids_train, ids_test, info = process_multiple_features(configs, column_names, random_state=42)
    """
    # Create working copies
    x_train, x_test, y_train, train_ids, test_ids = create_working_copies()

    info_dict = {}

    print(f"\n{'='*60}")
    print(f"PROCESSING {len(feature_configs)} FEATURES")
    print(f"{'='*60}")

    for feature_name, config in feature_configs.items():
        dist_info = preprocess_discrete_feature(
            x_train,
            x_test,
            feature_name=feature_name,
            column_names=column_names,
            simple_replacements=config.get("simple_replacements"),
            redistribute_config=config.get("redistribute_config"),
            random_state=random_state,
        )
        info_dict[feature_name] = dist_info

    print(f"\n{'='*60}")
    print(f"ALL FEATURES PROCESSED SUCCESSFULLY")
    print(f"{'='*60}\n")

    return x_train, x_test, y_train, train_ids, test_ids, info_dict


# %%
def conditional_redistribute_with_distribution(
    x_train_feature: np.ndarray,
    x_test_feature: np.ndarray,
    x_train_condition: np.ndarray,
    x_test_condition: np.ndarray,
    condition_value: float,
    values_to_redistribute: List[float],
    valid_range: tuple,
    random_state: Optional[int] = None,
) -> tuple:
    """
    Conditionally redistribute special values based on a condition feature using distribution.
    When condition is met, redistribute problematic values using the distribution of valid values
    in the training data (filtered by the same condition).

    Parameters
    ----------
    x_train_feature : np.ndarray
        Training feature data to be filled (1D array)
    x_test_feature : np.ndarray
        Test feature data to be filled (1D array)
    x_train_condition : np.ndarray
        Training condition feature (same length as x_train_feature)
    x_test_condition : np.ndarray
        Test condition feature (same length as x_test_feature)
    condition_value : float or callable
        Value/condition that triggers redistribution (e.g., 1 for employed, lambda x: x >= 65 for age)
    values_to_redistribute : list of float
        Values to be redistributed when condition is met (e.g., [77, 99, np.nan])
    valid_range : tuple
        (min, max) of valid values for the distribution calculation
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    tuple : (x_train_filled, x_test_filled, distribution_info)
        - x_train_filled: Training data with conditional redistribution applied
        - x_test_filled: Test data with conditional redistribution applied
        - distribution_info: Dict with distribution statistics

    Example
    -------
    >>> # Fill missing income based on employment status
    >>> # When employed (=1), redistribute 77,99,NaN using income distribution of employed people
    >>> income_train = np.array([50000, 77, 75000, 99, np.nan, 45000])
    >>> employment_train = np.array([1, 1, 1, 1, 1, 0])  # 1=employed, 0=unemployed
    >>> income_test = np.array([77, 60000, np.nan])
    >>> employment_test = np.array([1, 1, 1])
    >>>
    >>> train_filled, test_filled, info = conditional_redistribute_with_distribution(
    ...     income_train, income_test, employment_train, employment_test,
    ...     condition_value=1,  # When employed
    ...     values_to_redistribute=[77, 99, np.nan],
    ...     valid_range=(20000, 200000),  # Valid income range
    ...     random_state=42
    ... )
    """
    if random_state is not None:
        np.random.seed(random_state)

    x_train_copy = x_train_feature.copy()
    x_test_copy = x_test_feature.copy()
    x_train_cond_copy = x_train_condition.copy()
    x_test_cond_copy = x_test_condition.copy()

    min_val, max_val = valid_range

    # Step 1: Create condition masks
    if callable(condition_value):
        condition_mask_train = condition_value(x_train_cond_copy)
        condition_mask_test = condition_value(x_test_cond_copy)
        condition_description = f"function({condition_value})"
    else:
        if np.isnan(condition_value):
            condition_mask_train = np.isnan(x_train_cond_copy)
            condition_mask_test = np.isnan(x_test_cond_copy)
        else:
            condition_mask_train = x_train_cond_copy == condition_value
            condition_mask_test = x_test_cond_copy == condition_value
        condition_description = str(condition_value)

    # Step 2: Identify valid values in training data UNDER THE CONDITION
    valid_mask_train = condition_mask_train.copy()  # Start with condition mask

    # Filter out values to redistribute
    for special_val in values_to_redistribute:
        if np.isnan(special_val):
            valid_mask_train &= ~np.isnan(x_train_copy)
        else:
            valid_mask_train &= x_train_copy != special_val

    # Filter by valid range
    valid_mask_train &= (x_train_copy >= min_val) & (x_train_copy <= max_val)

    # Get valid values from training data under condition
    valid_train_values = x_train_copy[valid_mask_train]

    if len(valid_train_values) == 0:
        print(
            f"Warning: No valid values found in training data under condition = {condition_description}"
        )
        print("Falling back to unconditional distribution...")
        # Fallback: use all valid values regardless of condition
        valid_mask_train_fallback = np.ones(len(x_train_copy), dtype=bool)
        for special_val in values_to_redistribute:
            if np.isnan(special_val):
                valid_mask_train_fallback &= ~np.isnan(x_train_copy)
            else:
                valid_mask_train_fallback &= x_train_copy != special_val
        valid_mask_train_fallback &= (x_train_copy >= min_val) & (
            x_train_copy <= max_val
        )
        valid_train_values = x_train_copy[valid_mask_train_fallback]

        if len(valid_train_values) == 0:
            raise ValueError(
                "No valid values found even in unconditional distribution!"
            )

    # Step 3: Calculate distribution from conditional training data
    unique_values, counts = np.unique(valid_train_values, return_counts=True)
    probabilities = counts / counts.sum()

    print(f"Conditional distribution (condition = {condition_description}):")
    print(
        f"  Calculated from {len(valid_train_values)} valid conditional training samples"
    )
    print(
        f"  Condition matches - Train: {condition_mask_train.sum()}, Test: {condition_mask_test.sum()}"
    )
    print(
        f"  Valid values range: {unique_values.min():.2f} to {unique_values.max():.2f}"
    )
    print(f"  Number of unique valid values: {len(unique_values)}")

    # Step 4: Identify values to redistribute in TRAINING data (only under condition)
    redistribute_mask_train = condition_mask_train.copy()
    for special_val in values_to_redistribute:
        if np.isnan(special_val):
            redistribute_mask_train &= np.isnan(x_train_copy)
        else:
            redistribute_mask_train &= x_train_copy == special_val

    num_to_replace_train = redistribute_mask_train.sum()
    print(f"  Training: Redistributing {num_to_replace_train} values under condition")

    # Step 5: Redistribute in TRAINING data
    if num_to_replace_train > 0:
        replacements_train = np.random.choice(
            unique_values, size=num_to_replace_train, p=probabilities
        )
        x_train_copy[redistribute_mask_train] = replacements_train

    # Step 6: Identify values to redistribute in TEST data (only under condition)
    redistribute_mask_test = condition_mask_test.copy()
    for special_val in values_to_redistribute:
        if np.isnan(special_val):
            redistribute_mask_test &= np.isnan(x_test_copy)
        else:
            redistribute_mask_test &= x_test_copy == special_val

    num_to_replace_test = redistribute_mask_test.sum()
    print(f"  Test: Redistributing {num_to_replace_test} values under condition")

    # Step 7: Redistribute in TEST data using SAME conditional distribution
    if num_to_replace_test > 0:
        replacements_test = np.random.choice(
            unique_values, size=num_to_replace_test, p=probabilities
        )
        x_test_copy[redistribute_mask_test] = replacements_test

    # Distribution info
    distribution_info = {
        "condition_description": condition_description,
        "unique_values": unique_values,
        "probabilities": probabilities,
        "n_valid_conditional_samples": len(valid_train_values),
        "n_condition_matches_train": condition_mask_train.sum(),
        "n_condition_matches_test": condition_mask_test.sum(),
        "n_redistributed_train": num_to_replace_train,
        "n_redistributed_test": num_to_replace_test,
    }

    return x_train_copy, x_test_copy, distribution_info


# %%
def complex_multi_step_preprocessing(
    x_train: np.ndarray,
    x_test: np.ndarray,
    feature_name: str,
    column_names: List[str],
    steps_config: List[Dict],
    random_state: Optional[int] = None,
) -> Dict:
    """
    Apply complex multi-step preprocessing to a continuous/ordinal feature.
    Supports three types of steps for flexible data filling:

    1. 'simple_replacement': Direct value replacement (e.g., 88 → 0)
    2. 'conditional_fixed_fill': Fixed value when condition met (e.g., unemployed → 0 income)
    3. 'conditional_redistribute': Distribution-based when condition met (e.g., employed → income distribution)
    4. 'redistribute': Unconditional distribution-based filling

    Parameters
    ----------
    x_train : np.ndarray
        Training data (2D array) - MODIFIED IN-PLACE
    x_test : np.ndarray
        Test data (2D array) - MODIFIED IN-PLACE
    feature_name : str
        Name of the feature to process
    column_names : list of str
        List of all column names in the dataset
    steps_config : list of dict
        Ordered list of preprocessing steps:
        [
            {
                'step': 'simple_replacement',
                'replacements': {88: 0, 'nan': 0}
            },
            {
                'step': 'conditional_redistribute',
                'condition_column': 'EMPLOY1',
                'condition_value': 1,  # or lambda function
                'values_to_redistribute': [77, 99, np.nan],
                'valid_range': (20000, 200000)
            },
            {
                'step': 'conditional_fixed_fill',
                'condition_column': 'EMPLOY1',
                'condition_value': 2,  # unemployed
                'target_values': [77, 99, np.nan],
                'fill_value': 0  # no income
            },
            {
                'step': 'redistribute',
                'values_to_redistribute': [77, 99, np.nan],
                'valid_range': (1, 30)
            }
        ]
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict : step_results_info
        Dictionary with results from each step

    Example
    -------
    >>> # METVL11_ example: Exercise METs processing
    >>> # Step 1: For exercisers (EXERANY2=1), fill NaN using METs distribution
    >>> # Step 2: For non-exercisers (EXERANY2=2), fill NaN with 0
    >>>
    >>> x_train_copy = x_train.copy()
    >>> x_test_copy = x_test.copy()
    >>>
    >>> steps = [
    ...     {'step': 'conditional_redistribute', 'condition_column': 'EXERANY2',
    ...      'condition_value': 1, 'target_values': [np.nan], 'valid_range': (1, 100)},
    ...     {'step': 'conditional_fixed_fill', 'condition_column': 'EXERANY2',
    ...      'condition_value': 2, 'target_values': [np.nan], 'fill_value': 0}
    ... ]
    >>>
    >>> info = complex_multi_step_preprocessing(
    ...     x_train_copy, x_test_copy, 'METVL11_', column_names, steps, random_state=42
    ... )
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Get feature index
    if feature_name not in column_names:
        raise ValueError(f"Feature '{feature_name}' not found in column_names")

    feature_index = column_names.index(feature_name)

    print(
        f"\n=== Complex Multi-Step Processing: '{feature_name}' (index {feature_index}) ==="
    )
    print(f"Number of steps: {len(steps_config)}")

    # Extract feature columns (these are VIEWS, will be modified in-place)
    train_feature = x_train[:, feature_index]
    test_feature = x_test[:, feature_index]

    step_results = {}

    # Show initial state
    initial_train_nan = np.sum(np.isnan(train_feature))
    initial_test_nan = np.sum(np.isnan(test_feature))
    initial_train_unique = np.unique(train_feature[~np.isnan(train_feature)])
    initial_test_unique = np.unique(test_feature[~np.isnan(test_feature)])

    print(f"Initial state:")
    print(
        f"  Train: {len(train_feature)} values, {initial_train_nan} NaN, unique: {len(initial_train_unique)}"
    )
    print(
        f"  Test: {len(test_feature)} values, {initial_test_nan} NaN, unique: {len(initial_test_unique)}"
    )

    # Apply each step in order
    for step_num, step_config in enumerate(steps_config, 1):
        step_type = step_config["step"]

        print(f"\n--- Step {step_num}: {step_type} ---")

        if step_type == "simple_replacement":
            # Simple value replacements
            replacements = step_config["replacements"]
            print(f"Applying simple replacements: {replacements}")

            for old_val, new_val in replacements.items():
                if old_val == "nan" or (
                    isinstance(old_val, str) and old_val.lower() == "nan"
                ):
                    print(f"  Replacing NaN → {new_val}")
                    simple_value_replacement(train_feature, np.nan, new_val)
                    simple_value_replacement(test_feature, np.nan, new_val)
                else:
                    print(f"  Replacing {old_val} → {new_val}")
                    simple_value_replacement(train_feature, old_val, new_val)
                    simple_value_replacement(test_feature, old_val, new_val)

            step_results[f"step_{step_num}"] = {
                "type": "simple_replacement",
                "replacements": replacements,
            }

        elif step_type == "conditional_redistribute":
            # Conditional redistribution based on another feature
            condition_column = step_config["condition_column"]
            condition_value = step_config["condition_value"]
            values_to_redistribute = step_config["values_to_redistribute"]
            valid_range = step_config["valid_range"]

            if condition_column not in column_names:
                print(f"Error: Condition column '{condition_column}' not found")
                continue

            condition_index = column_names.index(condition_column)
            train_condition = x_train[:, condition_index]
            test_condition = x_test[:, condition_index]

            print(
                f"Conditional redistribution based on '{condition_column}' = {condition_value}"
            )
            print(f"Values to redistribute: {values_to_redistribute}")
            print(f"Valid range: {valid_range}")

            # Apply conditional redistribution
            train_feature_new, test_feature_new, dist_info = (
                conditional_redistribute_with_distribution(
                    train_feature,
                    test_feature,
                    train_condition,
                    test_condition,
                    condition_value=condition_value,
                    values_to_redistribute=values_to_redistribute,
                    valid_range=valid_range,
                    random_state=None,  # Don't reset seed for each step
                )
            )

            # Update the arrays
            x_train[:, feature_index] = train_feature_new
            x_test[:, feature_index] = test_feature_new

            # Update our views
            train_feature = x_train[:, feature_index]
            test_feature = x_test[:, feature_index]

            step_results[f"step_{step_num}"] = {
                "type": "conditional_redistribute",
                "condition_column": condition_column,
                "condition_value": condition_value,
                "distribution_info": dist_info,
            }

        elif step_type == "conditional_fixed_fill":
            # Conditional fixed value filling (like your conditional_imputation)
            condition_column = step_config["condition_column"]
            condition_value = step_config["condition_value"]
            target_values = step_config["target_values"]
            fill_value = step_config["fill_value"]

            if condition_column not in column_names:
                print(f"Error: Condition column '{condition_column}' not found")
                continue

            condition_index = column_names.index(condition_column)
            train_condition = x_train[:, condition_index]
            test_condition = x_test[:, condition_index]

            print(
                f"Conditional fixed fill based on '{condition_column}' = {condition_value}"
            )
            print(f"Target values: {target_values}")
            print(f"Fill value: {fill_value}")

            # Apply conditional fixed filling to training data
            if callable(condition_value):
                train_cond_mask = condition_value(train_condition)
                test_cond_mask = condition_value(test_condition)
                condition_description = f"function({condition_value})"
            else:
                if np.isnan(condition_value):
                    train_cond_mask = np.isnan(train_condition)
                    test_cond_mask = np.isnan(test_condition)
                else:
                    train_cond_mask = train_condition == condition_value
                    test_cond_mask = test_condition == condition_value
                condition_description = str(condition_value)

            # Create target masks for training data
            train_target_mask = np.zeros(len(train_feature), dtype=bool)
            for target_val in target_values:
                if np.isnan(target_val):
                    train_target_mask |= np.isnan(train_feature)
                else:
                    train_target_mask |= train_feature == target_val

            # Create target masks for test data
            test_target_mask = np.zeros(len(test_feature), dtype=bool)
            for target_val in target_values:
                if np.isnan(target_val):
                    test_target_mask |= np.isnan(test_feature)
                else:
                    test_target_mask |= test_feature == target_val

            # Apply conditional filling: target is in target_values AND condition is met
            train_fill_mask = train_target_mask & train_cond_mask
            test_fill_mask = test_target_mask & test_cond_mask

            num_filled_train = train_fill_mask.sum()
            num_filled_test = test_fill_mask.sum()

            print(f"  Train condition matches: {train_cond_mask.sum()}")
            print(f"  Test condition matches: {test_cond_mask.sum()}")
            print(f"  Train target matches: {train_target_mask.sum()}")
            print(f"  Test target matches: {test_target_mask.sum()}")
            print(f"  Train filled: {num_filled_train}")
            print(f"  Test filled: {num_filled_test}")

            # Fill the values
            train_feature[train_fill_mask] = fill_value
            test_feature[test_fill_mask] = fill_value

            step_results[f"step_{step_num}"] = {
                "type": "conditional_fixed_fill",
                "condition_column": condition_column,
                "condition_value": condition_value,
                "fill_value": fill_value,
                "n_filled_train": num_filled_train,
                "n_filled_test": num_filled_test,
            }

        elif step_type == "redistribute":
            # Unconditional redistribution
            values_to_redistribute = step_config["values_to_redistribute"]
            valid_range = step_config["valid_range"]

            print(f"Unconditional redistribution")
            print(f"Values to redistribute: {values_to_redistribute}")
            print(f"Valid range: {valid_range}")

            # Apply unconditional redistribution
            train_feature_new, test_feature_new, dist_info = (
                redistribute_with_distribution(
                    train_feature,
                    test_feature,
                    values_to_redistribute=values_to_redistribute,
                    valid_range=valid_range,
                    random_state=None,  # Don't reset seed for each step
                )
            )

            # Update the arrays
            x_train[:, feature_index] = train_feature_new
            x_test[:, feature_index] = test_feature_new

            # Update our views
            train_feature = x_train[:, feature_index]
            test_feature = x_test[:, feature_index]

            step_results[f"step_{step_num}"] = {
                "type": "redistribute",
                "distribution_info": dist_info,
            }

        else:
            print(f"Error: Unknown step type '{step_type}'")
            continue

        # Show step results
        step_train_nan = np.sum(np.isnan(train_feature))
        step_test_nan = np.sum(np.isnan(test_feature))
        step_train_unique = np.unique(train_feature[~np.isnan(train_feature)])
        step_test_unique = np.unique(test_feature[~np.isnan(test_feature)])

        print(f"After step {step_num}:")
        print(
            f"  Train: {step_train_nan} NaN remaining, {len(step_train_unique)} unique values"
        )
        print(
            f"  Test: {step_test_nan} NaN remaining, {len(step_test_unique)} unique values"
        )

    # Final results
    final_train_nan = np.sum(np.isnan(train_feature))
    final_test_nan = np.sum(np.isnan(test_feature))
    final_train_unique = np.unique(train_feature[~np.isnan(train_feature)])
    final_test_unique = np.unique(test_feature[~np.isnan(test_feature)])

    print(f"\n=== Final Results for '{feature_name}' ===")
    print(
        f"Train: {initial_train_nan} → {final_train_nan} NaN ({initial_train_nan - final_train_nan} fixed)"
    )
    print(
        f"Test: {initial_test_nan} → {final_test_nan} NaN ({initial_test_nan - final_test_nan} fixed)"
    )
    print(
        f"Train unique values: {len(initial_train_unique)} → {len(final_train_unique)}"
    )
    print(f"Test unique values: {len(initial_test_unique)} → {len(final_test_unique)}")

    return step_results


# %%
def process_conditional_features(
    conditional_configs: List[Dict],
    column_names: List[str],
    random_state: Optional[int] = None,
) -> tuple:
    """
    Process multiple features using conditional redistribution configurations.

    Parameters
    ----------
    conditional_configs : list of dict
        List of conditional redistribution configurations from mae_dictionnary.py
    column_names : list of str
        List of all column names in the dataset
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    tuple : (x_train_processed, x_test_processed, y_train, train_ids, test_ids, info_dict)
        Processed datasets and dictionary of redistribution info for each conditional rule

    Example
    -------
    >>> from mae_dictionnary import CONDITIONAL_FEATURE_CONFIGS
    >>> x_train, x_test, y_train, ids_train, ids_test, info = process_conditional_features(
    ...     CONDITIONAL_FEATURE_CONFIGS, column_names, random_state=42
    ... )
    """
    # Create working copies
    x_train, x_test, y_train, train_ids, test_ids = create_working_copies()

    info_dict = {}

    print(f"\n{'='*60}")
    print(f"PROCESSING {len(conditional_configs)} CONDITIONAL FEATURES")
    print(f"{'='*60}")

    for i, config in enumerate(conditional_configs):
        target_feature = config["target_feature"]
        condition_column = config["condition_column"]
        condition_value = config["condition_value"]
        values_to_redistribute = config["values_to_redistribute"]
        valid_range = config["valid_range"]

        print(
            f"\n--- Conditional Rule {i+1}: {target_feature} | {condition_column} = {condition_value} ---"
        )

        # Validate features exist
        if target_feature not in column_names:
            print(f"Error: Target feature '{target_feature}' not found, skipping...")
            continue
        if condition_column not in column_names:
            print(
                f"Error: Condition feature '{condition_column}' not found, skipping..."
            )
            continue

        # Get feature indices
        target_idx = column_names.index(target_feature)
        condition_idx = column_names.index(condition_column)

        # Extract feature data
        train_target = x_train[:, target_idx]
        test_target = x_test[:, target_idx]
        train_condition = x_train[:, condition_idx]
        test_condition = x_test[:, condition_idx]

        # Apply conditional redistribution
        try:
            train_filled, test_filled, dist_info = (
                conditional_redistribute_with_distribution(
                    train_target,
                    test_target,
                    train_condition,
                    test_condition,
                    condition_value=condition_value,
                    values_to_redistribute=values_to_redistribute,
                    valid_range=valid_range,
                    random_state=random_state,
                )
            )

            # Update the datasets
            x_train[:, target_idx] = train_filled
            x_test[:, target_idx] = test_filled

            # Store results
            rule_key = f"{target_feature}_{condition_column}_{i+1}"
            info_dict[rule_key] = {
                "target_feature": target_feature,
                "condition_column": condition_column,
                "condition_value": condition_value,
                "distribution_info": dist_info,
            }

        except Exception as e:
            print(f"Error processing conditional rule {i+1}: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"CONDITIONAL FEATURES PROCESSED SUCCESSFULLY")
    print(f"{'='*60}\n")

    return x_train, x_test, y_train, train_ids, test_ids, info_dict


# %%
def process_complex_engineering_features(
    complex_configs: Dict[str, List[Dict]],
    column_names: List[str],
    random_state: Optional[int] = None,
) -> tuple:
    """
    Process multiple features using complex multi-step engineering configurations.

    Parameters
    ----------
    complex_configs : dict
        Dictionary mapping feature names to their multi-step configurations from mae_dictionnary.py
    column_names : list of str
        List of all column names in the dataset
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    tuple : (x_train_processed, x_test_processed, y_train, train_ids, test_ids, info_dict)
        Processed datasets and dictionary of step results for each feature

    Example
    -------
    >>> from mae_dictionnary import COMPLEX_ENGINEERING_CONFIGS
    >>> x_train, x_test, y_train, ids_train, ids_test, info = process_complex_engineering_features(
    ...     COMPLEX_ENGINEERING_CONFIGS, column_names, random_state=42
    ... )
    """
    # Create working copies
    x_train, x_test, y_train, train_ids, test_ids = create_working_copies()

    info_dict = {}

    print(f"\n{'='*60}")
    print(f"COMPLEX ENGINEERING: {len(complex_configs)} FEATURES")
    print(f"{'='*60}")

    for feature_name, steps_config in complex_configs.items():
        print(f"\n=== Processing '{feature_name}' with {len(steps_config)} steps ===")

        try:
            step_results = complex_multi_step_preprocessing(
                x_train,
                x_test,
                feature_name,
                column_names,
                steps_config,
                random_state=random_state,
            )
            info_dict[feature_name] = step_results

        except Exception as e:
            print(f"Error processing complex feature '{feature_name}': {e}")
            continue

    print(f"\n{'='*60}")
    print(f"COMPLEX ENGINEERING COMPLETED SUCCESSFULLY")
    print(f"{'='*60}\n")

    return x_train, x_test, y_train, train_ids, test_ids, info_dict


# %%
def apply_mae_comprehensive_pipeline(
    column_names: List[str],
    random_state: Optional[int] = None,
    include_simple: bool = True,
    include_conditional: bool = True,
    include_complex: bool = True,
) -> tuple:
    """
    Apply Mae's complete data filling pipeline: simple + conditional + complex engineering.
    This is the equivalent of your apply_imputation_configurations() but for continuous/ordinal variables.

    Parameters
    ----------
    column_names : list of str
        List of all column names in the dataset
    random_state : int, optional
        Random seed for reproducibility
    include_simple : bool, default=True
        Whether to apply simple feature preprocessing from FEATURE_CONFIGS
    include_conditional : bool, default=True
        Whether to apply conditional redistribution from CONDITIONAL_FEATURE_CONFIGS
    include_complex : bool, default=True
        Whether to apply complex engineering from COMPLEX_ENGINEERING_CONFIGS

    Returns
    -------
    tuple : (x_train_final, x_test_final, y_train, train_ids, test_ids, combined_info)
        Final processed datasets and combined information dictionary

    Example
    -------
    >>> from data_filling import get_column_names_from_csv
    >>> column_names = get_column_names_from_csv('dataset/x_train.csv')
    >>>
    >>> x_train_filled, x_test_filled, y_train, train_ids, test_ids, info = apply_mae_comprehensive_pipeline(
    ...     column_names, random_state=42
    ... )
    """
    from configs.imputation_configs_cont import (
        FEATURE_CONFIGS,
        CONDITIONAL_FEATURE_CONFIGS,
        COMPLEX_ENGINEERING_CONFIGS,
    )

    if random_state is not None:
        np.random.seed(random_state)

    print(f"\n{'='*80}")
    print(f"MAE'S COMPREHENSIVE DATA FILLING PIPELINE")
    print(f"Focused on continuous/ordinal variables with:")
    print(f"  • Distribution-based conditional filling")
    print(f"  • Fixed-value conditional filling")
    print(f"  • Multi-step complex engineering")
    print(f"{'='*80}")
    print(f"Simple features: {include_simple} ({len(FEATURE_CONFIGS)} configs)")
    print(
        f"Conditional features: {include_conditional} ({len(CONDITIONAL_FEATURE_CONFIGS)} configs)"
    )
    print(
        f"Complex engineering: {include_complex} ({len(COMPLEX_ENGINEERING_CONFIGS)} configs)"
    )
    print(f"{'='*80}")

    # Start with fresh copies
    x_train, x_test, y_train, train_ids, test_ids = create_working_copies()
    combined_info = {}

    # Step 1: Simple feature preprocessing
    if include_simple and FEATURE_CONFIGS:
        print(f"\n🔸 STEP 1: SIMPLE FEATURE PREPROCESSING")
        try:
            x_train, x_test, y_train, train_ids, test_ids, simple_info = (
                process_multiple_features(
                    FEATURE_CONFIGS, column_names, random_state=random_state
                )
            )
            combined_info["simple_features"] = simple_info
            print(f"✓ Simple features completed: {len(simple_info)} features processed")
        except Exception as e:
            print(f"✗ Error in simple features: {e}")
            combined_info["simple_features"] = {"error": str(e)}

    # Step 2: Conditional redistribution
    if include_conditional and CONDITIONAL_FEATURE_CONFIGS:
        print(f"\n🔸 STEP 2: CONDITIONAL REDISTRIBUTION")
        try:
            x_train, x_test, y_train, train_ids, test_ids, conditional_info = (
                process_conditional_features(
                    CONDITIONAL_FEATURE_CONFIGS, column_names, random_state=random_state
                )
            )
            combined_info["conditional_features"] = conditional_info
            print(
                f"✓ Conditional features completed: {len(conditional_info)} rules processed"
            )
        except Exception as e:
            print(f"✗ Error in conditional features: {e}")
            combined_info["conditional_features"] = {"error": str(e)}

    # Step 3: Complex multi-step engineering
    if include_complex and COMPLEX_ENGINEERING_CONFIGS:
        print(f"\n🔸 STEP 3: COMPLEX MULTI-STEP ENGINEERING")
        try:
            x_train, x_test, y_train, train_ids, test_ids, complex_info = (
                process_complex_engineering_features(
                    COMPLEX_ENGINEERING_CONFIGS, column_names, random_state=random_state
                )
            )
            combined_info["complex_features"] = complex_info
            print(
                f"✓ Complex engineering completed: {len(complex_info)} features processed"
            )
        except Exception as e:
            print(f"✗ Error in complex features: {e}")
            combined_info["complex_features"] = {"error": str(e)}

    # Final summary
    print(f"\n{'='*80}")
    print(f"MAE'S PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"Final dataset shapes - Train: {x_train.shape}, Test: {x_test.shape}")

    # Check for remaining NaN values
    train_nan = np.sum(np.isnan(x_train))
    test_nan = np.sum(np.isnan(x_test))
    print(f"Remaining NaN values - Train: {train_nan:,}, Test: {test_nan:,}")

    if train_nan > 0 or test_nan > 0:
        print(
            f"⚠️  Some NaN values remain (this is expected for features not in configurations)"
        )
    else:
        print(f"✓ All configured features filled successfully!")

    print(f"{'='*80}\n")

    return x_train, x_test, y_train, train_ids, test_ids, combined_info


def perform_single_feature_imputation(
    feature_name: str,
    imputation_config: List[Dict],
    x_train: Optional[np.ndarray] = None,
    x_test: Optional[np.ndarray] = None,
    column_names: Optional[List[str]] = None,
    train_ids: Optional[np.ndarray] = None,
    test_ids: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
    save_results: bool = True,
    output_dir: str = "dataset_your_custom_pipeline/dataset_your_custom_final",
) -> Dict:
    """
    Perform imputation on a single feature using the specified configuration.
    Similar to test_metvl11_example but generalized for any feature.

    Parameters
    ----------
    feature_name : str
        Name of the feature to impute
    imputation_config : list of dict
        List of imputation steps configuration
    x_train : np.ndarray, optional
        Training data. If not provided, will be loaded from files
    x_test : np.ndarray, optional
        Test data. If not provided, will be loaded from files
    column_names : list of str, optional
        Column names. If not provided, will be loaded from files
    random_state : int, optional
        Random seed for reproducibility
    save_results : bool, default=True
        Whether to save the imputed datasets back to files

    Returns
    -------
    dict : step_results
        Dictionary with results from each imputation step
    """
    print("=" * 60)
    print(f"IMPUTING FEATURE: {feature_name}")
    print("=" * 60)

    # Load data and column names if not provided
    if x_train is None or x_test is None or column_names is None:
        try:
            # Use our existing helper function
            train_data, train_header = read_csv_with_header(
                "dataset_your_custom_pipeline/dataset_your_custom_final/x_train_filled.csv"
            )
            test_data, test_header = read_csv_with_header(
                "dataset_your_custom_pipeline/dataset_your_custom_final/x_test_filled.csv"
            )

            column_names = train_header[1:]  # Skip Id column
            train_ids = train_data[:, 0].astype(int)
            test_ids = test_data[:, 0].astype(int)
            x_train = train_data[:, 1:]  # Exclude Id column
            x_test = test_data[:, 1:]  # Exclude Id column
            print(f"✓ Data loaded: Train {x_train.shape}, Test {x_test.shape}")
        except Exception as e:
            print(f"❌ ERROR: Could not load data: {e}")
            return None

    # Check if feature exists
    if feature_name not in column_names:
        print(f"❌ ERROR: Feature '{feature_name}' not found in the dataset")
        return None

    # Get feature index
    feature_idx = column_names.index(feature_name)

    # Show initial state
    print(f"\n--- Initial State ---")
    train_feature = x_train[:, feature_idx]
    test_feature = x_test[:, feature_idx]

    train_nan = np.sum(np.isnan(train_feature))
    test_nan = np.sum(np.isnan(test_feature))

    print(f"{feature_name} - Train NaN: {train_nan}, Test NaN: {test_nan}")

    # Apply the complex multi-step preprocessing
    print(f"\n--- Applying Multi-Step Processing ---")

    try:
        step_results = complex_multi_step_preprocessing(
            x_train,
            x_test,
            feature_name,
            column_names,
            imputation_config,
            random_state=random_state,
        )

        # Show final results
        print(f"\n--- Final Results ---")
        final_train_feature = x_train[:, feature_idx]
        final_test_feature = x_test[:, feature_idx]

        final_train_nan = np.sum(np.isnan(final_train_feature))
        final_test_nan = np.sum(np.isnan(final_test_feature))

        print(
            f"{feature_name} - Final Train NaN: {final_train_nan}, Test NaN: {final_test_nan}"
        )

        success = final_train_nan == 0 and final_test_nan == 0
        print(f"\n{'✓ SUCCESS' if success else '❌ FAILED'}: Imputation completed!")

        # Save the results if requested
        if save_results:
            try:
                import os
                import shutil

                # Back up current files
                backup_dir = f"{output_dir}_backup"
                os.makedirs(backup_dir, exist_ok=True)
                shutil.copy2(
                    f"{output_dir}/x_train_filled.csv",
                    f"{backup_dir}/x_train_filled.csv",
                )
                shutil.copy2(
                    f"{output_dir}/x_test_filled.csv", f"{backup_dir}/x_test_filled.csv"
                )
                print(f"\n✓ Backed up existing files to {backup_dir}/")

                # Save datasets using numpy to avoid scientific notation
                os.makedirs(output_dir, exist_ok=True)

                # Save training set
                train_with_ids = np.column_stack((train_ids.reshape(-1, 1), x_train))
                header = "Id," + ",".join(column_names)
                np.savetxt(
                    f"{output_dir}/x_train_filled.csv",
                    train_with_ids,
                    fmt="%.6g",
                    delimiter=",",
                    header=header,
                    comments="",
                )

                # Save test set
                test_with_ids = np.column_stack((test_ids.reshape(-1, 1), x_test))
                np.savetxt(
                    f"{output_dir}/x_test_filled.csv",
                    test_with_ids,
                    fmt="%.6g",
                    delimiter=",",
                    header=header,
                    comments="",
                )

                print(f"\n✓ Saved imputed datasets to {output_dir}/")
            except Exception as e:
                print(f"\n❌ ERROR saving datasets: {e}")

        return step_results

    except Exception as e:
        print(f"❌ ERROR in processing: {e}")
        return None
