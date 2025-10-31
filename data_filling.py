"""
Data imputation and filling module for handling missing or invalid values in datasets.

This module provides a comprehensive set of functions for data imputation, including:
- Single feature imputation
- Conditional value imputation
- Weighted random filling
- Complex feature engineering
- Batch value replacement

The module supports both simple value replacement and sophisticated conditional
imputation based on feature relationships. It uses numpy for efficient array operations
and handles both training and test datasets consistently.

Main functions:
- apply_imputation_configurations: Main entry point for applying all imputation steps
- conditional_imputation_dataset: Impute values based on conditions in other features
- fill_dataset_with_column_config: Fill missing values using weighted random sampling
- apply_single_feature_imputation: Handle imputation for individual features
"""

import numpy as np
from typing import List, Optional, Dict, Union, Callable, Tuple
import os
import shutil
from helpers import load_csv_data
from configs.imputation_configs_OHE import complex_engineering_configs


def apply_single_feature_imputation_hivstd3(
    x_train: np.ndarray, x_test: np.ndarray, column_names: List[str], config: List[Dict]
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply imputation specifically for HIVTSTD3-like features that use:
    - weighted_random
    - conditional_weighted_random
    - conditional_fill
    """
    # Make copies of the data to avoid modifying the original
    x_train = x_train.copy()
    x_test = x_test.copy()

    feature_idx = column_names.index("feature_105")  # HIVTSTD3
    condition_idx = column_names.index("feature_104")  # HIVTST6

    train_feature = x_train[:, feature_idx]
    test_feature = x_test[:, feature_idx]
    train_condition = x_train[:, condition_idx]
    test_condition = x_test[:, condition_idx]

    # Apply each step
    for step in config:
        step_type = step["step"]
        print(f"\nApplying {step_type}...")

        if step_type == "weighted_random":
            # Replace special codes with weighted random values
            target_values = step["target_values"]
            valid_values = step["valid_values"]
            probabilities = step["probabilities"]

            # Create masks for values to replace
            train_mask = np.isin(train_feature, target_values)
            test_mask = np.isin(test_feature, target_values)

            # Sample from valid values with given probabilities
            train_feature[train_mask] = np.random.choice(
                valid_values, size=train_mask.sum(), p=probabilities
            )
            test_feature[test_mask] = np.random.choice(
                valid_values, size=test_mask.sum(), p=probabilities
            )

        elif step_type == "conditional_weighted_random":
            target_values = step["target_values"]
            condition_value = step["condition_value"]
            valid_values = step["valid_values"]
            probabilities = step["probabilities"]

            # Create condition masks
            train_cond_mask = train_condition == condition_value
            test_cond_mask = test_condition == condition_value

            # Create target masks (e.g., NaN values)
            train_target_mask = np.isin(train_feature, target_values)
            test_target_mask = np.isin(test_feature, target_values)

            # Combine masks - only fill where both condition is met and value needs filling
            train_fill_mask = train_cond_mask & train_target_mask
            test_fill_mask = test_cond_mask & test_target_mask

            # Fill with weighted random values
            train_feature[train_fill_mask] = np.random.choice(
                valid_values, size=train_fill_mask.sum(), p=probabilities
            )
            test_feature[test_fill_mask] = np.random.choice(
                valid_values, size=test_fill_mask.sum(), p=probabilities
            )

        elif step_type == "conditional_fill":
            target_values = step["target_values"]
            condition_value = step["condition_value"]
            fill_value = step["fill_value"]

            # Create condition masks
            train_cond_mask = train_condition == condition_value
            test_cond_mask = test_condition == condition_value

            # Create target masks
            train_target_mask = np.isin(train_feature, target_values)
            test_target_mask = np.isin(test_feature, target_values)

            # Combine masks
            train_fill_mask = train_cond_mask & train_target_mask
            test_fill_mask = test_cond_mask & test_target_mask

            # Fill with fixed value
            train_feature[train_fill_mask] = fill_value
            test_feature[test_fill_mask] = fill_value

    # Update the arrays with filled values
    x_train[:, feature_idx] = train_feature
    x_test[:, feature_idx] = test_feature

    return x_train, x_test


def conditional_redistribute_with_distribution(
    x_train_feature: np.ndarray,
    x_test_feature: np.ndarray,
    x_train_condition: np.ndarray,
    x_test_condition: np.ndarray,
    condition_value: Union[int, float, Callable],
    values_to_redistribute: List[float],
    valid_range: Tuple[float, float],
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Conditionally redistribute values in a feature based on condition from another feature.
    When condition is met, values are sampled from the distribution of valid values.

    Parameters
    ----------
    x_train_feature, x_test_feature : np.ndarray
        Features to fill
    x_train_condition, x_test_condition : np.ndarray
        Features containing condition values
    condition_value : Union[int, float, Callable]
        Value that defines when to redistribute
    values_to_redistribute : List[float]
        Values to be redistributed (e.g., [77, 99, np.nan])
    valid_range : Tuple[float, float]
        (min, max) of valid values for sampling
    random_state : int, optional
        Random seed

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Dict]
        Filled arrays and distribution info
    """
    if random_state is not None:
        np.random.seed(random_state)

    train_copy = x_train_feature.copy()
    test_copy = x_test_feature.copy()

    min_val, max_val = valid_range

    # Create condition masks
    if callable(condition_value):
        train_cond_mask = condition_value(x_train_condition)
        test_cond_mask = condition_value(x_test_condition)
        condition_desc = f"function({condition_value})"
    else:
        train_cond_mask = x_train_condition == condition_value
        test_cond_mask = x_test_condition == condition_value
        condition_desc = str(condition_value)

    # Get valid values under condition
    valid_mask_train = train_cond_mask.copy()

    # Filter out values to redistribute
    for val in values_to_redistribute:
        if np.isnan(val):
            valid_mask_train &= ~np.isnan(train_copy)
        else:
            valid_mask_train &= train_copy != val

    # Filter by valid range
    valid_mask_train &= (train_copy >= min_val) & (train_copy <= max_val)
    valid_values = train_copy[valid_mask_train]

    if len(valid_values) == 0:
        raise ValueError("No valid values found under condition for distribution")

    # Calculate distribution
    unique_vals, counts = np.unique(valid_values, return_counts=True)
    probs = counts / counts.sum()

    print(f"Conditional distribution under {condition_desc}:")
    print(f"  Using {len(valid_values)} valid values")
    print(f"  Range: {unique_vals.min():.2f} to {unique_vals.max():.2f}")
    print(f"  {len(unique_vals)} unique values")

    # Apply to training set
    train_fill_mask = train_cond_mask.copy()
    for val in values_to_redistribute:
        if np.isnan(val):
            train_fill_mask &= np.isnan(train_copy)
        else:
            train_fill_mask &= train_copy == val

    n_train_fill = train_fill_mask.sum()
    if n_train_fill > 0:
        train_copy[train_fill_mask] = np.random.choice(
            unique_vals, size=n_train_fill, p=probs
        )

    # Apply to test set
    test_fill_mask = test_cond_mask.copy()
    for val in values_to_redistribute:
        if np.isnan(val):
            test_fill_mask &= np.isnan(test_copy)
        else:
            test_fill_mask &= test_copy == val

    n_test_fill = test_fill_mask.sum()
    if n_test_fill > 0:
        test_copy[test_fill_mask] = np.random.choice(
            unique_vals, size=n_test_fill, p=probs
        )

    dist_info = {
        "condition_description": condition_desc,
        "unique_values": unique_vals,
        "probabilities": probs,
        "n_valid_samples": len(valid_values),
        "n_filled_train": n_train_fill,
        "n_filled_test": n_test_fill,
    }

    return train_copy, test_copy, dist_info


def _normalize_probabilities(
    values: List[str], probabilities: List[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Helper function to normalize probability weights."""
    values_array = np.array([float(v) for v in values])
    probs_array = np.array(probabilities)
    probs_array = probs_array / probs_array.sum()
    return values_array, probs_array


def fill_column_with_weights(
    column: np.ndarray, weights: Dict[str, float], random_state: Optional[int] = None
) -> np.ndarray:
    """
    Fill NaN values using weighted random sampling.

    Parameters
    ----------
    column : np.ndarray
        Column to fill
    weights : Dict[str, float]
        Dictionary mapping values to their weights
    random_state : int, optional
        Random seed

    Returns
    -------
    np.ndarray
        Filled column
    """
    if random_state is not None:
        np.random.seed(random_state)

    result = column.copy()
    nan_mask = np.isnan(result)

    if not nan_mask.any():
        return result

    # Normalize probabilities
    values, probs = _normalize_probabilities(
        list(weights.keys()), list(weights.values())
    )

    # Fill NaN values
    n_fill = nan_mask.sum()
    choices = np.random.choice(values, size=n_fill, p=probs)
    result[nan_mask] = choices

    return result


def apply_single_feature_imputation(
    feature_name: str,
    config: Dict,
    x_train: np.ndarray,
    x_test: np.ndarray,
    column_names: List[str],
    train_ids: Optional[np.ndarray] = None,
    test_ids: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
    input_dir: str = "dataset_your_custom_pipeline/dataset_your_custom_final_backup",
    output_dir: Optional[str] = None,
    save_results: bool = True,
) -> Dict:
    """
    Apply imputation for a single feature using configuration from imputation_configs.py.
    Supports all types of configurations: value replacements, conditional imputations,
    weighted random sampling.

    Parameters
    ----------
    feature_name : str
        Name of the feature to impute (e.g., 'GENHLTH', 'INCOME2')
    config : Dict
        The imputation configuration from imputation_configs.py.
        Can be from:
        - value_replacement_configs
        - conditional_imputation_configs
        - weighted_random_configs
    x_train : np.ndarray
        Training data array
    x_test : np.ndarray
        Test data array
    column_names : List[str]
        List of column names
    train_ids : np.ndarray, optional
        Training data IDs (needed if save_results=True)
    test_ids : np.ndarray, optional
        Test data IDs (needed if save_results=True)
    random_state : int, optional
        Random seed for reproducibility
    output_dir : str, optional
        Directory to save imputed datasets
    save_results : bool, default=True
        Whether to save results to files

    Returns
    -------
    Dict
        Dictionary with imputation results and statistics

    Example
    -------
    >>> from imputation_configs import value_replacement_configs
    >>> result = apply_single_feature_imputation(
    ...     'GENHLTH',
    ...     value_replacement_configs['GENHLTH'],
    ...     x_train,
    ...     x_test,
    ...     column_names,
    ...     train_ids,
    ...     test_ids
    ... )
    """
    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Get feature index
    if feature_name not in column_names:
        raise ValueError(f"Feature '{feature_name}' not found in column names")
    feature_idx = column_names.index(feature_name)

    print(f"\n{'='*60}")
    print(f"IMPUTING FEATURE: {feature_name}")

    # Check NaN values before imputation
    train_nan_before = np.sum(np.isnan(x_train[:, feature_idx]))
    test_nan_before = np.sum(np.isnan(x_test[:, feature_idx]))
    print(f"\nNaN values before imputation:")
    print(
        f"  Training set: {train_nan_before} ({train_nan_before/len(x_train)*100:.2f}%)"
    )
    print(f"  Test set: {test_nan_before} ({test_nan_before/len(x_test)*100:.2f}%)")

    # Initial state
    train_feature = x_train[:, feature_idx]
    test_feature = x_test[:, feature_idx]

    train_nan = np.sum(np.isnan(train_feature))
    test_nan = np.sum(np.isnan(test_feature))
    train_unique = np.unique(train_feature[~np.isnan(train_feature)])

    print(f"\nInitial state:")
    print(f"  Train NaN: {train_nan}, Test NaN: {test_nan}")
    print(f"  Train unique values: {sorted(train_unique)}")

    imputation_info = {"feature_name": feature_name}

    try:
        # Handle value replacements
        if "replacements" in config:
            print("\nApplying value replacements...")
            # Convert replacements list to dictionary for processing
            replacements = {str(old): float(new) for old, new in config["replacements"]}
            imputation_info["type"] = "value_replacement"

            # Apply replacements
            x_train[:, feature_idx] = replace_values_numpy(train_feature, replacements)
            x_test[:, feature_idx] = replace_values_numpy(test_feature, replacements)

            imputation_info["replacements"] = replacements

        # Handle conditional imputation
        elif "condition_feature" in config:
            print("\nApplying conditional imputation...")
            condition_feature = config["condition_feature"]
            condition_value = config["condition_value"]
            condition_idx = column_names.index(condition_feature)

            # Get condition arrays
            train_condition = x_train[:, condition_idx]
            test_condition = x_test[:, condition_idx]

            if "target_values" in config:
                # Create target masks for values to impute
                train_mask = np.zeros(len(train_feature), dtype=bool)
                test_mask = np.zeros(len(test_feature), dtype=bool)

                for val in config["target_values"]:
                    if np.isnan(val):
                        train_mask |= np.isnan(train_feature)
                        test_mask |= np.isnan(test_feature)
                    else:
                        train_mask |= train_feature == val
                        test_mask |= test_feature == val

                # Apply conditional fill
                condition_mask_train = train_condition == condition_value
                condition_mask_test = test_condition == condition_value

                fill_mask_train = train_mask & condition_mask_train
                fill_mask_test = test_mask & condition_mask_test

                fill_value = config["fill_value"]
                x_train[fill_mask_train, feature_idx] = fill_value
                x_test[fill_mask_test, feature_idx] = fill_value

                imputation_info.update(
                    {
                        "type": "conditional_fill",
                        "condition_feature": condition_feature,
                        "condition_value": condition_value,
                        "target_values": config["target_values"],
                        "fill_value": fill_value,
                        "n_filled_train": int(fill_mask_train.sum()),
                        "n_filled_test": int(fill_mask_test.sum()),
                    }
                )

            elif "distribution_range" in config:
                # Distribution-based conditional imputation
                valid_range = config["distribution_range"]
                values_to_redistribute = config["values_to_redistribute"]

                (
                    train_feature_new,
                    test_feature_new,
                    dist_info,
                ) = conditional_redistribute_with_distribution(
                    train_feature,
                    test_feature,
                    train_condition,
                    test_condition,
                    condition_value=condition_value,
                    values_to_redistribute=values_to_redistribute,
                    valid_range=valid_range,
                    random_state=random_state,
                )

                x_train[:, feature_idx] = train_feature_new
                x_test[:, feature_idx] = test_feature_new

                imputation_info.update(
                    {
                        "type": "conditional_distribute",
                        "condition_feature": condition_feature,
                        "condition_value": condition_value,
                        "distribution_info": dist_info,
                    }
                )

        # Handle weighted random sampling
        elif "weights" in config:
            print("\nApplying weighted random sampling...")
            weights = config["weights"]
            imputation_info["type"] = "weighted_random"

            # Apply weighted random fill
            x_train[:, feature_idx] = fill_column_with_weights(
                train_feature, weights, random_state=random_state
            )
            x_test[:, feature_idx] = fill_column_with_weights(
                test_feature, weights, random_state=random_state
            )

            imputation_info["weights"] = weights

        else:
            raise ValueError(
                f"Unknown imputation configuration type for feature {feature_name}"
            )

        # Show final results
        final_train_feature = x_train[:, feature_idx]
        final_test_feature = x_test[:, feature_idx]

        final_train_nan = np.sum(np.isnan(final_train_feature))
        final_test_nan = np.sum(np.isnan(final_test_feature))
        final_train_unique = np.unique(
            final_train_feature[~np.isnan(final_train_feature)]
        )

        print(f"\n--- Final Results ---")
        print(f"NaN values after imputation:")
        print(
            f"  Training set: {final_train_nan} → {final_train_nan/len(x_train)*100:.2f}% (was {train_nan_before} → {train_nan_before/len(x_train)*100:.2f}%)"
        )
        print(
            f"  Test set: {final_test_nan} → {final_test_nan/len(x_test)*100:.2f}% (was {test_nan_before} → {test_nan_before/len(x_test)*100:.2f}%)"
        )
        print(f"Final train unique values: {sorted(final_train_unique)}")

        # Print change summary
        print(f"\nImputation Summary:")
        print(
            f"  Training set NaNs removed: {train_nan_before - final_train_nan} ({(train_nan_before - final_train_nan)/train_nan_before*100:.2f}% reduction)"
        )
        if test_nan_before > 0:
            print(
                f"  Test set NaNs removed: {test_nan_before - final_test_nan} ({(test_nan_before - final_test_nan)/test_nan_before*100:.2f}% reduction)"
            )

        success = final_train_nan == 0 and final_test_nan == 0
        print(f"\n{'✓ SUCCESS' if success else '❌ WARNING'}: Imputation completed!")

        imputation_info["results"] = {
            "initial_nan": {"train": int(train_nan), "test": int(test_nan)},
            "final_nan": {"train": int(final_train_nan), "test": int(final_test_nan)},
            "success": success,
        }

        # Save results if requested
        if save_results:
            if train_ids is None or test_ids is None:
                raise ValueError(
                    "train_ids and test_ids are required when save_results=True"
                )

            try:
                # Determine output directory
                save_dir = output_dir if output_dir is not None else input_dir
                os.makedirs(save_dir, exist_ok=True)

                # Save datasets using numpy
                # Save training set
                train_with_ids = np.column_stack((train_ids.reshape(-1, 1), x_train))
                header = "Id," + ",".join(column_names)
                np.savetxt(
                    f"{save_dir}/x_train_filled.csv",
                    train_with_ids,
                    fmt="%.6g",
                    delimiter=",",
                    header=header,
                    comments="",
                )

                # Save test set
                test_with_ids = np.column_stack((test_ids.reshape(-1, 1), x_test))
                np.savetxt(
                    f"{save_dir}/x_test_filled.csv",
                    test_with_ids,
                    fmt="%.6g",
                    delimiter=",",
                    header=header,
                    comments="",
                )

                print(f"✓ Saved imputed datasets to {save_dir}/")
                imputation_info["saved"] = True

            except Exception as e:
                print(f"\n❌ ERROR saving datasets: {e}")
                imputation_info["saved"] = False
                imputation_info["save_error"] = str(e)

        return imputation_info

    except Exception as e:
        print(f"❌ ERROR during imputation: {e}")
        imputation_info["error"] = str(e)
        return imputation_info


def _create_missing_mask(data: np.ndarray, missing_values: List[float]) -> np.ndarray:
    """Helper function to create a mask for missing values."""
    missing_mask = np.zeros(len(data), dtype=bool)
    for missing_val in missing_values:
        if np.isnan(missing_val):
            missing_mask |= np.isnan(data)
        else:
            missing_mask |= data == missing_val
    return missing_mask


def weighted_random_fill_numpy(
    data: np.ndarray,
    missing_values: List[float],
    valid_values: Optional[List[float]] = None,
    probabilities: Optional[List[float]] = None,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Replace missing values with weighted random choices using provided or calculated probabilities.
    Numpy-only implementation.

    Parameters:
    -----------
    data : np.ndarray
        The feature data containing missing values to be filled (1D array)
    missing_values : list
        List of values to be considered as missing (e.g., [9, np.nan] for "don't know" and blank)
    valid_values : list, optional
        List of valid values to use for replacement. Required if probabilities are provided.
        If None and probabilities is None, all values except missing_values are considered valid
    probabilities : list of float, optional
        Pre-computed probabilities for each valid value (must sum to 1.0).
        If None, probabilities are calculated from existing data distribution.
        Length must match valid_values.
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    np.ndarray
        Data with missing values replaced by weighted random choices

    Examples:
    ---------
    >>> # Using pre-computed probabilities
    >>> feature = np.array([1, 2, 7, 9, 1, 2, np.nan, 7, 1])
    >>> filled_feature = weighted_random_fill_numpy(feature, missing_values=[9, np.nan],
    ...                                            valid_values=[1, 2, 7], probabilities=[0.2858, 0.5, 0.2142])

    >>> # Auto-calculate probabilities from data
    >>> filled_feature = weighted_random_fill_numpy(feature, missing_values=[9, np.nan], valid_values=[1, 2, 7])
    """

    if random_state is not None:
        np.random.seed(random_state)

    # Work on a copy to avoid modifying original data
    data_array = data.copy()

    # Create mask for missing values
    missing_mask = _create_missing_mask(data_array, missing_values)

    # If no missing values, return original data
    if not missing_mask.any():
        return data

    # User provided probabilities
    if valid_values is None:
        raise ValueError(
            "valid_values must be provided when probabilities are specified"
        )

    if len(probabilities) != len(valid_values):
        raise ValueError("Length of probabilities must match length of valid_values")

    # Normalize probabilities to ensure they sum to 1
    probabilities = np.array(probabilities)
    if not np.isclose(probabilities.sum(), 1.0, rtol=1e-5):
        print(
            f"Warning: Probabilities sum to {probabilities.sum():.6f}, normalizing to 1.0"
        )
        probabilities = probabilities / probabilities.sum()

    unique_values = np.array(valid_values)

    # Generate random replacements
    num_missing = missing_mask.sum()
    replacements = np.random.choice(unique_values, size=num_missing, p=probabilities)

    # Fill missing values
    data_array[missing_mask] = replacements

    return data_array


def conditional_imputation_numpy(
    target_data: np.ndarray,
    condition_data: np.ndarray,
    target_missing_values: List[float],
    condition_value: float,
    fill_value: float,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Conditionally impute missing values in target feature based on condition feature value.

    Parameters:
    -----------
    target_data : np.ndarray
        The feature data to be imputed (1D array)
    condition_data : np.ndarray
        The condition feature data to check (1D array, same length as target_data)
    target_missing_values : list
        List of values considered as missing in target_data (e.g., [9, np.nan])
    condition_value : float
        Value in condition_data that triggers the conditional imputation
    fill_value : float
        Value to use for imputation when condition is met
    random_state : int, optional
        Random seed for reproducibility (for consistency with other functions)

    Returns:
    --------
    np.ndarray
        Target data with conditional imputation applied

    Examples:
    ---------
    >>> # Fill missing values in 'income' with 0 when 'employment_status' is 0 (unemployed)
    >>> income = np.array([50000, np.nan, 75000, 9, 60000])
    >>> employment = np.array([1, 0, 1, 0, 1])  # 0=unemployed, 1=employed
    >>> filled_income = conditional_imputation_numpy(
    ...     target_data=income,
    ...     condition_data=employment,
    ...     target_missing_values=[9, np.nan],
    ...     condition_value=0,  # When unemployed
    ...     fill_value=0        # Fill with 0 income
    ... )
    >>> # Result: [50000, 0, 75000, 0, 60000]

    >>> # Fill missing values in 'insurance' with 2 (no insurance) when 'age' < 18
    >>> insurance = np.array([1, np.nan, 1, 9, 2])
    >>> age = np.array([25, 16, 30, 15, 40])
    >>> filled_insurance = conditional_imputation_numpy(
    ...     target_data=insurance,
    ...     condition_data=age,
    ...     target_missing_values=[9, np.nan],
    ...     condition_value=lambda x: x < 18,  # Custom condition function
    ...     fill_value=2
    ... )
    """

    if random_state is not None:
        np.random.seed(random_state)

    # Convert to numpy arrays and make copies
    target_array = np.array(target_data, dtype=float).copy()
    condition_array = np.array(condition_data, dtype=float).copy()

    if len(target_array) != len(condition_array):
        raise ValueError(
            f"Target and condition arrays must have same length: {len(target_array)} vs {len(condition_array)}"
        )

    # Create missing value mask for target data
    target_missing_mask = np.zeros(len(target_array), dtype=bool)
    for missing_val in target_missing_values:
        if np.isnan(missing_val):
            target_missing_mask |= np.isnan(target_array)
        else:
            target_missing_mask |= target_array == missing_val

    # Create condition mask
    if callable(condition_value):
        # Handle function-based conditions (like lambda x: x < 18)
        condition_mask = condition_value(condition_array)
    else:
        # Handle exact value matching
        if np.isnan(condition_value):
            condition_mask = np.isnan(condition_array)
        else:
            condition_mask = condition_array == condition_value

    # Apply conditional imputation: target is missing AND condition is met
    imputation_mask = target_missing_mask & condition_mask

    print(f"Conditional imputation:")
    print(f"  Target missing values: {np.sum(target_missing_mask)} locations")
    print(f"  Condition matches: {np.sum(condition_mask)} locations")
    print(f"  Conditional imputation applied: {np.sum(imputation_mask)} locations")
    print(f"  Fill value: {fill_value}")

    # Fill the values
    target_array[imputation_mask] = fill_value

    return target_array


def conditional_weighted_random_fill_numpy(
    target_data: np.ndarray,
    condition_data: np.ndarray,
    target_missing_values: List[float],
    condition_value: float,
    valid_values: List[float],
    probabilities: List[float],
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Conditionally impute missing values using weighted random selection based on condition.

    This combines conditional logic with probability distributions - when a condition is met,
    missing values are filled using weighted random choice from specified values/probabilities.

    Parameters:
    -----------
    target_data : np.ndarray
        The feature data to be imputed (1D array)
    condition_data : np.ndarray
        The condition feature data to check (1D array, same length as target_data)
    target_missing_values : list
        List of values considered as missing in target_data (e.g., [7, 9, np.nan])
    condition_value : float or callable
        Value/condition in condition_data that triggers the imputation
    valid_values : list
        List of valid values to choose from for imputation
    probabilities : list
        Corresponding probabilities for each valid value (must sum to 1.0)
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    np.ndarray
        Target data with conditional weighted random imputation applied

    Examples:
    ---------
    >>> # Fill missing DIABETE3 values based on AGE with different probabilities
    >>> diabete3 = np.array([1, 7, 2, 9, 1])  # 1=yes, 2=no, 7/9=missing
    >>> age = np.array([65, 70, 30, 75, 45])
    >>>
    >>> # When age >= 65, fill missing with weighted probabilities
    >>> filled_diabete3 = conditional_weighted_random_fill_numpy(
    ...     target_data=diabete3,
    ...     condition_data=age,
    ...     target_missing_values=[7, 9],
    ...     condition_value=lambda x: x >= 65,  # Condition: age >= 65
    ...     valid_values=[1, 2],               # Yes or No diabetes
    ...     probabilities=[0.7, 0.3],          # 70% yes, 30% no for elderly
    ...     random_state=42
    ... )
    >>> # Missing values at indices 1,3 will be filled since ages 70,75 >= 65

    >>> # Different example: Income based on employment status
    >>> income = np.array([50000, np.nan, 75000, 9, 60000])
    >>> employment = np.array([1, 1, 1, 1, 0])  # 1=employed, 0=unemployed
    >>>
    >>> filled_income = conditional_weighted_random_fill_numpy(
    ...     target_data=income,
    ...     condition_data=employment,
    ...     target_missing_values=[9, np.nan],
    ...     condition_value=1,  # When employed
    ...     valid_values=[30000, 50000, 70000, 100000],
    ...     probabilities=[0.2, 0.4, 0.3, 0.1],  # Income distribution for employed
    ...     random_state=42
    ... )
    """

    if random_state is not None:
        np.random.seed(random_state)

    # Normalize probabilities and validate inputs
    valid_values_array, normalized_probs = _normalize_probabilities(
        list(map(str, valid_values)), probabilities
    )

    if not np.isclose(np.sum(probabilities), 1.0, atol=1e-6):
        raise ValueError(f"Probabilities must sum to 1.0, got: {np.sum(probabilities)}")

    # Convert to numpy arrays and make copies
    target_array = np.array(target_data, dtype=float).copy()
    condition_array = np.array(condition_data, dtype=float).copy()

    if len(target_array) != len(condition_array):
        raise ValueError(
            f"Target and condition arrays must have same length: {len(target_array)} vs {len(condition_array)}"
        )

    # Create missing value mask for target data
    target_missing_mask = np.zeros(len(target_array), dtype=bool)
    for missing_val in target_missing_values:
        if np.isnan(missing_val):
            target_missing_mask |= np.isnan(target_array)
        else:
            target_missing_mask |= target_array == missing_val

    # Create condition mask
    if callable(condition_value):
        # Handle function-based conditions (like lambda x: x >= 65)
        condition_mask = condition_value(condition_array)
    else:
        # Handle exact value matching
        if np.isnan(condition_value):
            condition_mask = np.isnan(condition_array)
        else:
            condition_mask = condition_array == condition_value

    # Apply conditional weighted random imputation: target is missing AND condition is met
    imputation_mask = target_missing_mask & condition_mask
    num_to_fill = np.sum(imputation_mask)

    if num_to_fill > 0:
        # Generate weighted random values
        filled_values = np.random.choice(
            valid_values, size=num_to_fill, p=probabilities
        )

        # Fill the values
        target_array[imputation_mask] = filled_values

    print(f"Conditional weighted random imputation:")
    print(f"  Target missing values: {np.sum(target_missing_mask)} locations")
    print(f"  Condition matches: {np.sum(condition_mask)} locations")
    print(f"  Conditional imputation applied: {num_to_fill} locations")
    print(f"  Values: {valid_values} with probabilities: {probabilities}")

    return target_array


def multi_conditional_imputation_numpy(
    target_data: np.ndarray,
    condition_data_dict: Dict[str, np.ndarray],
    target_missing_values: List[float],
    condition_rules: Dict[str, float],
    fill_value: float,
    logical_operator: str = "AND",
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Conditionally impute missing values based on multiple condition features.

    Parameters:
    -----------
    target_data : np.ndarray
        The feature data to be imputed (1D array)
    condition_data_dict : dict
        Dictionary mapping condition column names to their data arrays
        e.g., {'CVDINFR4': array1, 'CVDCRHD4': array2}
    target_missing_values : list
        List of values considered as missing in target_data (e.g., [9, np.nan])
    condition_rules : dict
        Dictionary mapping condition column names to required values
        e.g., {'CVDINFR4': 2, 'CVDCRHD4': 2}
    fill_value : float
        Value to use for imputation when conditions are met
    logical_operator : str, default='AND'
        How to combine multiple conditions: 'AND' or 'OR'
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    np.ndarray
        Target data with multi-conditional imputation applied

    Examples:
    ---------
    >>> # Fill _MICHD with 2 (no heart disease) when both CVDINFR4=2 AND CVDCRHD4=2
    >>> michd = np.array([1, np.nan, 2, 9, 1])
    >>> cvdinfr4 = np.array([1, 2, 2, 2, 1])  # 1=yes MI, 2=no MI
    >>> cvdcrhd4 = np.array([2, 2, 1, 2, 2])  # 1=yes CHD, 2=no CHD
    >>>
    >>> filled_michd = multi_conditional_imputation_numpy(
    ...     target_data=michd,
    ...     condition_data_dict={'CVDINFR4': cvdinfr4, 'CVDCRHD4': cvdcrhd4},
    ...     target_missing_values=[9, np.nan],
    ...     condition_rules={'CVDINFR4': 2, 'CVDCRHD4': 2},  # Both no MI AND no CHD
    ...     fill_value=2,  # Fill with 2 (no heart disease)
    ...     logical_operator='AND'
    ... )
    >>> # Result: [1, 2, 2, 2, 1] - fills index 1 and 3 where conditions met
    """

    if random_state is not None:
        np.random.seed(random_state)

    # Convert to numpy array and make copy
    target_array = np.array(target_data, dtype=float).copy()

    # Validate all condition arrays have same length as target
    for col_name, condition_array in condition_data_dict.items():
        condition_array = np.array(condition_array, dtype=float)
        if len(target_array) != len(condition_array):
            raise ValueError(
                f"Target and condition array '{col_name}' must have same length: {len(target_array)} vs {len(condition_array)}"
            )

    # Create missing value mask for target data
    target_missing_mask = np.zeros(len(target_array), dtype=bool)
    for missing_val in target_missing_values:
        if np.isnan(missing_val):
            target_missing_mask |= np.isnan(target_array)
        else:
            target_missing_mask |= target_array == missing_val

    # Create individual condition masks
    condition_masks = {}
    for col_name, required_value in condition_rules.items():
        if col_name not in condition_data_dict:
            raise ValueError(
                f"Condition column '{col_name}' not found in condition_data_dict"
            )

        condition_array = np.array(condition_data_dict[col_name], dtype=float)

        if callable(required_value):
            # Handle function-based conditions
            condition_masks[col_name] = required_value(condition_array)
        else:
            # Handle exact value matching
            if np.isnan(required_value):
                condition_masks[col_name] = np.isnan(condition_array)
            else:
                condition_masks[col_name] = condition_array == required_value

    # Combine condition masks using logical operator
    if logical_operator.upper() == "AND":
        combined_condition_mask = np.ones(len(target_array), dtype=bool)
        for col_name, mask in condition_masks.items():
            combined_condition_mask &= mask
    elif logical_operator.upper() == "OR":
        combined_condition_mask = np.zeros(len(target_array), dtype=bool)
        for col_name, mask in condition_masks.items():
            combined_condition_mask |= mask
    else:
        raise ValueError(
            f"logical_operator must be 'AND' or 'OR', got '{logical_operator}'"
        )

    # Apply conditional imputation: target is missing AND conditions are met
    imputation_mask = target_missing_mask & combined_condition_mask

    print(f"Multi-conditional imputation:")
    print(f"  Target missing values: {np.sum(target_missing_mask)} locations")

    # Show individual condition matches
    for col_name, mask in condition_masks.items():
        required_val = condition_rules[col_name]
        print(f"  {col_name} = {required_val}: {np.sum(mask)} matches")

    print(
        f"  Combined conditions ({logical_operator}): {np.sum(combined_condition_mask)} matches"
    )
    print(
        f"  Multi-conditional imputation applied: {np.sum(imputation_mask)} locations"
    )
    print(f"  Fill value: {fill_value}")

    # Fill the values
    target_array[imputation_mask] = fill_value

    return target_array


def fill_dataset_with_column_config(
    x_data: np.ndarray,
    column_names: List[str],
    column_configs: Dict[str, Dict],
    default_missing_values: Optional[List[float]] = None,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Fill missing values in dataset features using column names and weighted random filling.

    Parameters:
    -----------
    x_data : np.ndarray
        The dataset (2D array) where rows are samples and columns are features
    column_names : list of str
        List of column names corresponding to the features in x_data
    column_configs : dict
        Dictionary mapping column names to their filling configuration:
        {
            'column_name': {
                'valid_values': [1, 2, 7],
                'probabilities': [0.2858, 0.5, 0.2142],
                'missing_values': [9, np.nan]  # Optional: specific missing values for this column
            }
        }
    default_missing_values : list, optional
        Default list of values to be considered as missing if not specified in column config
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    np.ndarray
        Dataset with missing values filled

    Example:
    --------
    >>> column_names = ['age', 'income', 'education', 'satisfaction']
    >>> configs = {
    ...     'satisfaction': {
    ...         'valid_values': [1, 2, 7],
    ...         'probabilities': [0.2858, 0.5, 0.2142]
    ...     }
    ... }
    >>> x_filled = fill_dataset_with_column_config(x_data, column_names, configs)
    """

    if random_state is not None:
        np.random.seed(random_state)

    if len(column_names) != x_data.shape[1]:
        raise ValueError(
            f"Number of column names ({len(column_names)}) must match number of features ({x_data.shape[1]})"
        )

    # Work on a copy to avoid modifying original data
    x_filled = x_data.copy()

    # Reduced verbose output - keeping only essential info
    print(f"📊 Processing {len(column_configs)} features on dataset {x_filled.shape}")
    # print(f"Available columns: {column_names}")  # Too verbose
    # print(f"Columns to fill: {list(column_configs.keys())}")  # Already shown elsewhere

    for column_name, config in column_configs.items():
        if column_name not in column_names:
            print(
                f"Warning: Column '{column_name}' not found in column_names, skipping..."
            )
            continue

        # Get column index
        column_idx = column_names.index(column_name)

        print(f"\n--- Filling column '{column_name}' (index {column_idx}) ---")

        # Extract feature column
        feature_data = x_filled[:, column_idx]

        # Get configuration for weighted random filling
        valid_values = config.get("valid_values", None)
        probabilities = config.get("probabilities", None)
        column_missing_values = config.get("missing_values", default_missing_values)

        if column_missing_values is None:
            raise ValueError(
                f"Missing values must be specified either in column config for '{column_name}' or as default_missing_values parameter"
            )

        print(f"Using method: weighted_random")
        print(f"Valid values: {valid_values}")
        print(f"Probabilities: {probabilities}")
        print(f"Missing values: {column_missing_values}")

        # Fill the feature using weighted random filling
        filled_feature = weighted_random_fill_numpy(
            feature_data,
            missing_values=column_missing_values,
            valid_values=valid_values,
            probabilities=probabilities,
            random_state=None,  # Don't reset seed for each feature
        )

        # Update the dataset
        x_filled[:, column_idx] = filled_feature

    return x_filled


def conditional_imputation_dataset(
    x_data: np.ndarray,
    column_names: List[str],
    conditional_configs: List[Dict],
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Apply conditional imputation rules to multiple features in a dataset.
    Supports both single-condition and multi-condition rules.

    Parameters:
    -----------
    x_data : np.ndarray
        The dataset (2D array: rows=samples, columns=features)
    column_names : list of str
        List of column names corresponding to features
    conditional_configs : list of dict
        List of conditional imputation configurations. Each config can be:

        Single condition:
        {
            'target_column': 'column_name_to_fill',
            'condition_column': 'column_name_to_check',
            'target_missing_values': [9, np.nan],
            'condition_value': value_or_function,
            'fill_value': value_to_fill_with
        }

        Multi-condition:
        {
            'target_column': 'column_name_to_fill',
            'condition_columns': ['col1', 'col2'],  # Multiple condition columns
            'target_missing_values': [9, np.nan],
            'condition_rules': {'col1': 2, 'col2': 2},  # Rules for each column
            'logical_operator': 'AND',  # 'AND' or 'OR'
            'fill_value': value_to_fill_with
        }

    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    np.ndarray
        Dataset with conditional imputation applied

    Examples:
    ---------
    >>> # Single condition rules
    >>> conditional_rules = [
    ...     {
    ...         'target_column': 'income',
    ...         'condition_column': 'employment_status',
    ...         'target_missing_values': [9, np.nan],
    ...         'condition_value': 0,  # unemployed
    ...         'fill_value': 0       # no income
    ...     },
    ...     # Multi-condition rule
    ...     {
    ...         'target_column': '_MICHD',
    ...         'condition_columns': ['CVDINFR4', 'CVDCRHD4'],
    ...         'target_missing_values': [9, np.nan],
    ...         'condition_rules': {'CVDINFR4': 2, 'CVDCRHD4': 2},  # Both no heart issues
    ...         'logical_operator': 'AND',
    ...         'fill_value': 2  # No heart disease
    ...     }
    ... ]
    >>>
    >>> filled_data = conditional_imputation_dataset(
    ...     x_data, column_names, conditional_rules, random_state=42
    ... )
    """

    if random_state is not None:
        np.random.seed(random_state)

    if len(column_names) != x_data.shape[1]:
        raise ValueError(
            f"Number of column names ({len(column_names)}) must match number of features ({x_data.shape[1]})"
        )

    # Work on a copy to avoid modifying original data
    x_filled = x_data.copy()

    print(f"Applying conditional imputation to dataset with shape {x_filled.shape}")
    print(f"Number of conditional rules: {len(conditional_configs)}")

    for i, config in enumerate(conditional_configs):
        print(f"\n--- Conditional Rule {i+1} ---")

        # Extract common configuration
        target_col = config["target_column"]
        target_missing_values = config["target_missing_values"]
        fill_value = config["fill_value"]

        # Get target column index
        try:
            target_idx = column_names.index(target_col)
        except ValueError as e:
            print(f"Error: Target column '{target_col}' not found - {e}")
            continue

        target_data = x_filled[:, target_idx]

        # Check if this is a single-condition or multi-condition rule
        if "condition_column" in config:
            # Single condition rule
            condition_col = config["condition_column"]
            condition_value = config["condition_value"]

            try:
                condition_idx = column_names.index(condition_col)
            except ValueError as e:
                print(f"Error: Condition column '{condition_col}' not found - {e}")
                continue

            print(f"Target: '{target_col}' (index {target_idx})")
            print(f"Single condition: '{condition_col}' (index {condition_idx})")

            condition_data = x_filled[:, condition_idx]

            filled_target = conditional_imputation_numpy(
                target_data=target_data,
                condition_data=condition_data,
                target_missing_values=target_missing_values,
                condition_value=condition_value,
                fill_value=fill_value,
                random_state=None,
            )

        elif "condition_columns" in config:
            # Multi-condition rule
            condition_columns = config["condition_columns"]
            condition_rules = config["condition_rules"]
            logical_operator = config.get("logical_operator", "AND")

            print(f"Target: '{target_col}' (index {target_idx})")
            print(f"Multi-conditions: {condition_columns}")
            print(f"Rules: {condition_rules}")
            print(f"Operator: {logical_operator}")

            # Build condition data dictionary
            condition_data_dict = {}
            missing_columns = []

            for col in condition_columns:
                try:
                    col_idx = column_names.index(col)
                    condition_data_dict[col] = x_filled[:, col_idx]
                except ValueError:
                    missing_columns.append(col)

            if missing_columns:
                print(f"Error: Condition columns not found: {missing_columns}")
                continue

            filled_target = multi_conditional_imputation_numpy(
                target_data=target_data,
                condition_data_dict=condition_data_dict,
                target_missing_values=target_missing_values,
                condition_rules=condition_rules,
                fill_value=fill_value,
                logical_operator=logical_operator,
                random_state=None,
            )

        else:
            print(
                f"Error: Rule must have either 'condition_column' or 'condition_columns'"
            )
            continue

        # Update the dataset
        x_filled[:, target_idx] = filled_target

    print(f"\nConditional imputation completed!")
    return x_filled


def complex_feature_engineering(
    x_data: np.ndarray,
    column_names: List[str],
    engineering_configs: Dict[str, List[Dict]],
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Apply complex multi-step feature engineering transformations to dataset features.
    Each feature can have multiple ordered transformation steps.

    Parameters:
    -----------
    x_data : np.ndarray
        The dataset (2D array: rows=samples, columns=features)
    column_names : list of str
        List of column names corresponding to features
    engineering_configs : dict
        Dictionary mapping column names to ordered lists of transformation steps:
        {
            'column_name': [
                {
                    'step': 'weighted_random',
                    'target_values': [7, 9],
                    'valid_values': [1, 2, 3],
                    'probabilities': [0.3, 0.5, 0.2]
                },
                {
                    'step': 'conditional_fill',
                    'target_values': [np.nan],
                    'condition_column': 'other_column',
                    'condition_value': 2,
                    'fill_value': 3
                },
                {
                    'step': 'weighted_random',
                    'target_values': [np.nan],
                    'valid_values': [1, 2, 3],
                    'probabilities': [0.4, 0.4, 0.2]
                }
            ]
        }
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    np.ndarray
        Dataset with complex feature engineering applied

    Examples:
    ---------
    >>> # SMOKDAY2 example: 3-step transformation
    >>> engineering_configs = {
    ...     'SMOKDAY2': [
    ...         {
    ...             'step': 'weighted_random',
    ...             'target_values': [7, 9],  # Replace 7 and 9 first
    ...             'valid_values': [1, 2, 3],
    ...             'probabilities': [0.2, 0.6, 0.2]
    ...         },
    ...         {
    ...             'step': 'conditional_fill',
    ...             'target_values': [np.nan],  # Then replace NaN conditionally
    ...             'condition_column': 'SMOKE100',
    ...             'condition_value': 2,  # When SMOKE100 = 2 (never smoked)
    ...             'fill_value': 3       # Fill with 3 (not at all)
    ...         },
    ...         {
    ...             'step': 'conditional_weighted_fill',
    ...             'target_values': [np.nan],           # Remaining NaN values
    ...             'condition_column': '_AGE80',        # Based on age
    ...             'condition_value': lambda x: x >= 65, # When age >= 65
    ...             'valid_values': [1, 2],             # Different probabilities
    ...             'probabilities': [0.8, 0.2]          # 80% daily, 20% some days
    ...         },
    ...         {
    ...             'step': 'weighted_random',
    ...             'target_values': [np.nan],  # Finally, remaining NaN for younger
    ...             'valid_values': [1, 2, 3],
    ...             'probabilities': [0.3, 0.5, 0.2]
    ...         }
    ...     ]
    ... }
    """

    if random_state is not None:
        np.random.seed(random_state)

    if len(column_names) != x_data.shape[1]:
        raise ValueError(
            f"Number of column names ({len(column_names)}) must match number of features ({x_data.shape[1]})"
        )

    # Work on a copy to avoid modifying original data
    x_filled = x_data.copy()

    print(
        f"Applying complex feature engineering to dataset with shape {x_filled.shape}"
    )
    print(f"Features to engineer: {list(engineering_configs.keys())}")

    for column_name, steps in engineering_configs.items():
        print(f"\n--- Engineering '{column_name}' with {len(steps)} steps ---")

        try:
            column_idx = column_names.index(column_name)
        except ValueError:
            print(f"Error: Column '{column_name}' not found in column_names")
            continue

        print(f"Column index: {column_idx}")
        feature_data = x_filled[:, column_idx].copy()

        # Show initial state
        initial_missing = np.sum(np.isnan(feature_data))
        initial_unique = np.unique(feature_data[~np.isnan(feature_data)])
        print(
            f"Initial state: {len(feature_data)} values, {initial_missing} NaN, unique values: {initial_unique}"
        )

        # Apply each step in order
        for step_num, step_config in enumerate(steps, 1):
            step_type = step_config["step"]
            target_values = step_config["target_values"]

            print(f"\n  Step {step_num}: {step_type}")
            print(f"  Target values to transform: {target_values}")

            # Count how many values will be affected
            target_mask = np.zeros(len(feature_data), dtype=bool)
            for target_val in target_values:
                if np.isnan(target_val):
                    target_mask |= np.isnan(feature_data)
                else:
                    target_mask |= feature_data == target_val

            values_to_transform = np.sum(target_mask)
            print(f"  Values to transform: {values_to_transform}")

            if values_to_transform == 0:
                print(f"  No values to transform, skipping step")
                continue

            # Apply the transformation step
            if step_type == "weighted_random":
                valid_values = step_config["valid_values"]
                probabilities = step_config["probabilities"]

                # Apply weighted random filling only to target values
                feature_data = weighted_random_fill_numpy(
                    feature_data,
                    missing_values=target_values,
                    valid_values=valid_values,
                    probabilities=probabilities,
                    random_state=None,  # Don't reset seed for each step
                )
                print(
                    f"  Applied weighted random: {valid_values} with prob {probabilities}"
                )

            elif step_type == "conditional_fill":
                condition_column = step_config["condition_column"]
                condition_value = step_config["condition_value"]
                fill_value = step_config["fill_value"]

                try:
                    condition_idx = column_names.index(condition_column)
                    condition_data = x_filled[:, condition_idx]

                    # Apply conditional imputation only to target values
                    feature_data = conditional_imputation_numpy(
                        target_data=feature_data,
                        condition_data=condition_data,
                        target_missing_values=target_values,
                        condition_value=condition_value,
                        fill_value=fill_value,
                        random_state=None,
                    )
                    print(
                        f"  Applied conditional fill: when '{condition_column}' = {condition_value}, fill with {fill_value}"
                    )

                except ValueError:
                    print(f"  Error: Condition column '{condition_column}' not found")
                    continue

            elif step_type == "conditional_weighted_fill":
                condition_column = step_config["condition_column"]
                condition_value = step_config["condition_value"]
                valid_values = step_config["valid_values"]
                probabilities = step_config["probabilities"]

                try:
                    condition_idx = column_names.index(condition_column)
                    condition_data = x_filled[:, condition_idx]

                    # Apply conditional weighted random imputation
                    feature_data = conditional_weighted_random_fill_numpy(
                        target_data=feature_data,
                        condition_data=condition_data,
                        target_missing_values=target_values,
                        condition_value=condition_value,
                        valid_values=valid_values,
                        probabilities=probabilities,
                        random_state=None,
                    )
                    print(
                        f"  Applied conditional weighted fill: when '{condition_column}' = {condition_value}"
                    )
                    print(
                        f"    Values: {valid_values} with probabilities: {probabilities}"
                    )

                except ValueError:
                    print(f"  Error: Condition column '{condition_column}' not found")
                    continue

            elif step_type == "fixed_fill":
                fill_value = step_config["fill_value"]

                # Fill all target values with fixed value
                for target_val in target_values:
                    if np.isnan(target_val):
                        feature_data[np.isnan(feature_data)] = fill_value
                    else:
                        feature_data[feature_data == target_val] = fill_value

                print(f"  Applied fixed fill: {target_values} -> {fill_value}")

            else:
                print(f"  Error: Unknown step type '{step_type}'")
                continue

            # Show step results
            remaining_missing = np.sum(np.isnan(feature_data))
            remaining_unique = np.unique(feature_data[~np.isnan(feature_data)])
            print(
                f"  After step: {remaining_missing} NaN remaining, unique values: {remaining_unique}"
            )

        # Update the dataset
        x_filled[:, column_idx] = feature_data

        # Show final results for this feature
        final_missing = np.sum(np.isnan(feature_data))
        final_unique = np.unique(feature_data[~np.isnan(feature_data)])
        print(
            f"\nFinal '{column_name}': {final_missing} NaN, unique values: {final_unique}"
        )

    print(f"\nComplex feature engineering completed!")
    return x_filled


def replace_values_numpy(data, column_names, target_column, old_value, new_value):
    """
    Replace all instances of a specific value with another value in a given column.

    This is a simple direct replacement function - useful for recoding values,
    fixing data entry errors, or standardizing codes.

    Parameters:
    -----------
    data : numpy.ndarray
        2D array where each row is an observation and each column is a feature
    column_names : list
        List of column names corresponding to data columns
    target_column : str
        Name of the column to perform replacement in
    old_value : float or int
        The value to be replaced (can be np.nan for NaN values)
    new_value : float or int
        The value to replace with

    Returns:
    --------
    numpy.ndarray
        Data array with values replaced

    Example:
    --------
    # Replace all 9s with NaN in the SMOKE100 column
    data = replace_values_numpy(data, column_names, 'SMOKE100', 9, np.nan)

    # Replace all 7s with 2s in HIVTST6 column
    data = replace_values_numpy(data, column_names, 'HIVTST6', 7, 2)
    """

    # Make a copy to avoid modifying original data
    data = data.copy()

    # Get column index
    if target_column not in column_names:
        print(f"Warning: Column '{target_column}' not found in column_names")
        return data

    col_idx = column_names.index(target_column)

    # Handle NaN replacement specially
    if np.isnan(old_value):
        # Replace NaN values
        mask = np.isnan(data[:, col_idx])
    else:
        # Replace specific numeric values
        mask = data[:, col_idx] == old_value

    # Perform replacement
    data[mask, col_idx] = new_value

    print(
        f"Replaced {np.sum(mask):,} instances of {old_value} with {new_value} in column '{target_column}'"
    )

    return data


def batch_replace_values_numpy(data, column_names, replacement_configs):
    """
    Perform multiple value replacements across different columns using a configuration dictionary.

    Parameters:
    -----------
    data : numpy.ndarray
        2D array where each row is an observation and each column is a feature
    column_names : list
        List of column names corresponding to data columns
    replacement_configs : dict
        Dictionary mapping column names to replacement rules
        Format: {
            'column_name': {
                'replacements': [(old_value1, new_value1), (old_value2, new_value2), ...]
            }
        }

    Returns:
    --------
    numpy.ndarray
        Data array with all specified replacements performed

    Example:
    --------
    replacement_configs = {
        'HIVTST6': {
            'replacements': [(7, 2), (9, np.nan)]  # Replace 7->2, 9->NaN
        },
        'SMOKE100': {
            'replacements': [(9, np.nan)]  # Replace 9->NaN
        }
    }
    data = batch_replace_values_numpy(data, column_names, replacement_configs)
    """

    # Make a copy to avoid modifying original data
    data = data.copy()

    for column_name, config in replacement_configs.items():
        if column_name not in column_names:
            print(
                f"Warning: Column '{column_name}' not found in column_names, skipping..."
            )
            continue

        replacements = config.get("replacements", [])

        for old_value, new_value in replacements:
            data = replace_values_numpy(
                data, column_names, column_name, old_value, new_value
            )

    return data


def load_fill_and_save_data(
    data_path: str,
    column_names: List[str],
    column_configs: Dict[str, Dict],
    default_missing_values: Optional[List[float]] = None,
    output_dir: str = "dataset_filled",
    sub_sample: bool = False,
    random_state: Optional[int] = None,
):
    """
    Complete pipeline: Load dataset, fill missing values using column configurations, and save to CSV.

    Parameters:
    -----------
    data_path : str
        Path to the dataset folder
    column_names : list of str
        List of column names corresponding to the features in the dataset
    column_configs : dict
        Dictionary mapping column names to their filling configuration:
        {
            'column_name': {
                'valid_values': [1, 2, 7],
                'probabilities': [0.2858, 0.5, 0.2142],
                'missing_values': [9, np.nan]  # Optional, per-feature missing values
            }
        }
    default_missing_values : list, optional
        Default list of values to be considered as missing (e.g., [9, np.nan]).
        Used if a column configuration doesn't specify 'missing_values'.
    output_dir : str, default="dataset_filled"
        Directory where filled CSV files will be saved
    sub_sample : bool, default=False
        If True, subsample the data (passed to load_csv_data)
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    tuple
        (x_train_filled, x_test_filled, y_train, train_ids, test_ids)
        Where x_train_filled and x_test_filled have missing values replaced

    Example:
    --------
    >>> # Define column names and their filling strategies
    >>> column_names = ['age', 'income', 'education', 'satisfaction', 'health']
    >>> column_configs = {
    ...     'satisfaction': {
    ...         'valid_values': [1, 2, 7],
    ...         'probabilities': [0.2858, 0.5, 0.2142],  # Your 28.58% example
    ...         'missing_values': [9, np.nan]  # Specific missing values for this feature
    ...     },
    ...     'health': {
    ...         'valid_values': [0, 1, 2],
    ...         'probabilities': [0.3, 0.4, 0.3],
    ...         'missing_values': [9]  # Different missing values for this feature
    ...     }
    ... }
    >>>
    >>> x_train_filled, x_test_filled, y_train, train_ids, test_ids = load_fill_and_save_data(
    ...     'dataset/', column_names, column_configs,
    ...     default_missing_values=[9, np.nan],  # Default for columns without specific missing_values
    ...     output_dir='dataset_filled', random_state=42
    ... )
    """

    print("Loading original dataset...")
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(
        data_path, sub_sample=sub_sample
    )

    print(f"Original dataset shapes - Train: {x_train.shape}, Test: {x_test.shape}")

    # Fill training data
    print("\n" + "=" * 50)
    print("FILLING TRAINING DATA")
    print("=" * 50)
    x_train_filled = fill_dataset_with_column_config(
        x_train,
        column_configs,
        column_names,
        default_missing_values=default_missing_values,
        random_state=random_state,
    )

    # Fill test data with same configuration
    print("\n" + "=" * 50)
    print("FILLING TEST DATA")
    print("=" * 50)
    x_test_filled = fill_dataset_with_column_config(
        x_test,
        column_configs,
        column_names,
        default_missing_values=default_missing_values,
        random_state=random_state,  # Use same seed for consistency
    )

    print(f"\nData filling completed!")
    print(
        f"Filled dataset shapes - Train: {x_train_filled.shape}, Test: {x_test_filled.shape}"
    )

    # Save the filled dataset to CSV files
    save_filled_dataset(
        x_train_filled, x_test_filled, y_train, train_ids, test_ids, output_dir
    )

    return x_train_filled, x_test_filled, y_train, train_ids, test_ids


def save_filled_dataset(
    x_train_filled: np.ndarray,
    x_test_filled: np.ndarray,
    y_train: np.ndarray,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    output_dir: str = "dataset_filled",
):
    """
    Save the filled dataset to CSV files in the same format as the original.

    Parameters:
    -----------
    x_train_filled, x_test_filled : np.ndarray
        The filled datasets
    y_train, train_ids, test_ids : np.ndarray
        Original labels and IDs
    output_dir : str
        Directory to save the filled datasets
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data with IDs (same format as original)
    x_train_with_ids = np.column_stack([train_ids, x_train_filled])
    x_test_with_ids = np.column_stack([test_ids, x_test_filled])
    y_train_with_ids = np.column_stack([train_ids, y_train])

    # Save files with clear "_filled" naming
    np.savetxt(
        os.path.join(output_dir, "x_train_filled.csv"),
        x_train_with_ids,
        delimiter=",",
        header="Id,"
        + ",".join([f"feature_{i}" for i in range(x_train_filled.shape[1])]),
        comments="",
    )

    np.savetxt(
        os.path.join(output_dir, "x_test_filled.csv"),
        x_test_with_ids,
        delimiter=",",
        header="Id,"
        + ",".join([f"feature_{i}" for i in range(x_test_filled.shape[1])]),
        comments="",
    )

    np.savetxt(
        os.path.join(output_dir, "y_train.csv"),
        y_train_with_ids,
        delimiter=",",
        header="Id,Prediction",
        comments="",
        fmt="%d",
    )

    print(f"Filled dataset saved to '{output_dir}/' directory")
    print(f"Files: x_train_filled.csv, x_test_filled.csv, y_train.csv")


def get_column_names_from_csv(csv_file_path: str) -> List[str]:
    """
    Extract column names from CSV file header using pure Python (no pandas).
    Removes the 'Id' column to match load_csv_data behavior.

    Parameters:
    -----------
    csv_file_path : str
        Path to the CSV file

    Returns:
    --------
    List[str]
        List of column names (excluding 'Id' column)

    Example:
    --------
    >>> column_names = get_column_names_from_csv('dataset/x_train.csv')
    >>> print(f"First 5 columns: {column_names[:5]}")
    """
    with open(csv_file_path, "r") as file:
        header_line = file.readline().strip()
        all_columns = header_line.split(",")

        # Remove the 'Id' column to match load_csv_data behavior
        # load_csv_data does: x_train = x_train[:, 1:] (removes first column)
        if all_columns[0].lower() in ["id", "Id", "ID"]:
            column_names = all_columns[1:]  # Remove Id column
        else:
            column_names = all_columns

    print(f"Extracted {len(column_names)} column names from CSV (Id column removed)")
    print(f"First 5 columns: {column_names[:5]}")
    print(f"Last 5 columns: {column_names[-5:]}")

    return column_names


def apply_imputation_configurations(
    x_data: np.ndarray, column_names: List[str], random_state: Optional[int] = None
) -> np.ndarray:
    """
    Apply all imputation configurations to a dataset in the correct order.
    This is a unified function that applies the 4 types of imputation from imputation_configs.py.

    Order of operations:
    1. Value replacement (batch_replace_values_numpy)
    2. Conditional imputation (conditional_imputation_dataset)
    3. Weighted random imputation (fill_dataset_with_column_config)
    4. Complex feature engineering (complex_feature_engineering)

    Parameters:
    -----------
    x_data : np.ndarray
        Input dataset (2D array: rows=samples, columns=features)
    column_names : List[str]
        List of column names corresponding to features (must match x_data.shape[1])
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    np.ndarray
        Dataset with all imputations applied

    Example:
    --------
    >>> # Get column names
    >>> column_names = get_column_names_from_csv('dataset/x_train.csv')
    >>>
    >>> # Load data
    >>> x_train, x_test, y_train, train_ids, test_ids = load_csv_data('dataset/')
    >>>
    >>> # Apply imputation to training set
    >>> x_train_filled = apply_imputation_configurations(x_train, column_names, random_state=42)
    >>>
    >>> # Apply imputation to test set
    >>> x_test_filled = apply_imputation_configurations(x_test, column_names, random_state=42)
    """

    # Import configurations
    try:
        from configs.imputation_configs_OHE import (
            value_replacement_configs,
            conditional_imputation_configs,
            weighted_random_configs,
            complex_engineering_configs,
        )
    except ImportError as e:
        print(f"Error importing imputation configurations: {e}")
        print("Make sure configs/imputation_configs_OHE.py exists in the project root")
        return x_data.copy()

    if random_state is not None:
        np.random.seed(random_state)

    # Validate inputs
    if len(column_names) != x_data.shape[1]:
        raise ValueError(
            f"Number of column names ({len(column_names)}) must match number of features ({x_data.shape[1]})"
        )

    # Reduced verbose output
    print(f"⚙️  Applying imputation to dataset {x_data.shape}")

    # Start with a copy of the data
    x_filled = x_data.copy()

    # Step 1: Value Replacement
    print(f"\n--- Step 1: Value Replacement ---")
    try:
        x_filled = batch_replace_values_numpy(
            x_filled, column_names, value_replacement_configs
        )
        print(f"✓ Value replacement completed")
    except Exception as e:
        print(f"✗ Error in value replacement: {e}")

    # Step 2: Conditional Imputation
    print(f"\n--- Step 2: Conditional Imputation ---")
    try:
        x_filled = conditional_imputation_dataset(
            x_filled, column_names, conditional_imputation_configs, random_state=None
        )
        print(f"✓ Conditional imputation completed")
    except Exception as e:
        print(f"✗ Error in conditional imputation: {e}")

    # Step 3: Weighted Random Imputation
    print(f"\n--- Step 3: Weighted Random Imputation ---")
    try:
        x_filled = fill_dataset_with_column_config(
            x_filled,
            column_names,
            weighted_random_configs,
            default_missing_values=[9, np.nan],
            random_state=None,
        )
        print(f"✓ Weighted random imputation completed")
    except Exception as e:
        print(f"✗ Error in weighted random imputation: {e}")

    # Step 4: Complex Feature Engineering
    print(f"\n--- Step 4: Complex Feature Engineering ---")
    try:
        x_filled = complex_feature_engineering(
            x_filled, column_names, complex_engineering_configs, random_state=None
        )
        print(f"✓ Complex feature engineering completed")
    except Exception as e:
        print(f"✗ Error in complex feature engineering: {e}")

    print(f"\n=== IMPUTATION COMPLETED ===")
    print(f"Final dataset shape: {x_filled.shape}")

    # Check for remaining missing values
    total_missing = np.sum(np.isnan(x_filled))
    if total_missing > 0:
        print(f"⚠️  Warning: {total_missing} missing values remain in the dataset")
        missing_per_column = np.sum(np.isnan(x_filled), axis=0)
        columns_with_missing = np.where(missing_per_column > 0)[0]
        print(f"Columns with missing values: {len(columns_with_missing)}")
        for col_idx in columns_with_missing[:10]:  # Show first 10
            col_name = (
                column_names[col_idx]
                if col_idx < len(column_names)
                else f"column_{col_idx}"
            )
            print(f"  {col_name}: {missing_per_column[col_idx]} missing values")
    else:
        print(f"✓ No missing values remaining!")

    return x_filled
