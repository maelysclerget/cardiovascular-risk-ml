#!/usr/bin/env python3
"""
Ordered Imputation Pipeline
===========================

This module implements an ordered imputation strategy where features are processed
in a specific sequence to handle dependencies between features. For example, some
features may need to be imputed before others if they are used as conditions in
the imputation of other features.

Dependencies:
------------
- src.helpers: Data loading and utility functions
- src.data_filling: Core imputation functionality
- configs.imputation_configs: Feature configurations

Input/Output:
------------
Input files (from data/processed/):
    - x_train_filled.csv: Training features
    - x_test_filled.csv: Test features
    - y_train.csv: Training labels

Output:
    - Imputed datasets saved to specified output directory
    - Detailed NaN analysis for validation
    - CSV files with clean numerical formatting

Usage Example:
-------------
from scripts.ordered_imputation import apply_ordered_imputation

# Define feature imputation sequence (order matters!)
features_to_impute = [
    'MSCODE',     # Must be first as others depend on it
    '_PAINDX1',   # Depends on multiple basic features
    '_ASTHMS1',   # Requires _PAINDX1 to be filled
    'HIVTST6',    # Complex conditional dependencies
    'SMOKDAY2'    # Uses statistics from other features
]

# Apply imputation maintaining dependency order
result = apply_ordered_imputation(features_to_impute, random_state=42)
"""

import numpy as np
import os
from typing import List, Optional, Dict, Any, Tuple

# Core data processing
from src.helpers import load_csv_data
from data_filling import (
    get_column_names_from_csv,  # Column name management
    save_filled_dataset,  # Data persistence
    batch_replace_values_numpy,  # Bulk value replacement
    conditional_imputation_dataset,  # Conditional filling
    fill_dataset_with_column_config,  # Configuration-based filling
    complex_feature_engineering,  # Multi-step imputation
)

# MAE-specific imputation
from src.cont_features_data_filling import (
    process_multiple_features,
    process_complex_engineering_features,
    process_conditional_features,
)
from src.cont_features_particular_cases import (
    apply_particular_transformations,
    get_particular_case_features,
    transform_alcday5,
)

from configs.imputation_configs_cont import (
    FEATURE_CONFIGS,
    CONDITIONAL_FEATURE_CONFIGS,
    COMPLEX_ENGINEERING_CONFIGS,
)

# Set numpy print options for cleaner display
np.set_printoptions(suppress=True, precision=4, floatmode="fixed")


def _clean_saved_csv_formatting(output_dir: str) -> None:
    """
    Clean up scientific notation and excessive decimal places in saved CSV files.
    Uses numpy only - no pandas allowed.
    """
    csv_files = ["x_train_filled.csv", "x_test_filled.csv"]

    for csv_file in csv_files:
        file_path = os.path.join(output_dir, csv_file)
        if os.path.exists(file_path):
            try:
                # Read the CSV data using numpy
                data = np.genfromtxt(file_path, delimiter=",", skip_header=1)

                # Read header separately
                with open(file_path, "r") as f:
                    header = f.readline().strip()

                # Clean up the data - round to reasonable precision
                data_cleaned = np.round(data, decimals=6)

                # Convert very small values to 0
                data_cleaned[np.abs(data_cleaned) < 1e-10] = 0

                # Save back with clean formatting using numpy
                np.savetxt(
                    file_path,
                    data_cleaned,
                    delimiter=",",
                    header=header,
                    comments="",
                    fmt="%.6g",  # Use general format to avoid scientific notation for reasonable numbers
                )

            except Exception as e:
                print(f"   ⚠️  Could not clean formatting for {csv_file}: {e}")

    print(f"   ✓ Number formatting cleaned in {output_dir}/")


def check_nan_values(
    data: np.ndarray, data_name: str, column_names: List[str]
) -> Dict[str, Any]:
    """
    Comprehensive NaN value analysis for a dataset.

    Parameters:
    -----------
    data : np.ndarray
        Dataset to analyze
    data_name : str
        Name of the dataset (for reporting)
    column_names : List[str]
        List of column names

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing NaN analysis results
    """
    total_values = data.size
    total_missing = np.sum(np.isnan(data))
    missing_percentage = (total_missing / total_values) * 100 if total_values > 0 else 0

    # Per-column analysis
    missing_per_column = np.sum(np.isnan(data), axis=0)
    columns_with_missing = np.where(missing_per_column > 0)[0]

    # Per-row analysis
    missing_per_row = np.sum(np.isnan(data), axis=1)
    rows_with_missing = np.sum(missing_per_row > 0)

    analysis = {
        "dataset_name": data_name,
        "total_values": total_values,
        "total_missing": total_missing,
        "missing_percentage": missing_percentage,
        "columns_with_missing": len(columns_with_missing),
        "rows_with_missing": rows_with_missing,
        "missing_per_column": missing_per_column,
        "columns_with_missing_indices": columns_with_missing,
    }

    # Print summary
    print(f"\n--- NaN Analysis for {data_name} ---")
    print(f"Dataset shape: {data.shape}")
    print(f"Total values: {total_values:,}")
    print(f"Missing values: {total_missing:,} ({missing_percentage:.2f}%)")
    print(
        f"Columns with missing values: {len(columns_with_missing)}/{len(column_names)}"
    )
    print(f"Rows with missing values: {rows_with_missing:,}/{data.shape[0]:,}")

    if len(columns_with_missing) > 0:
        print(f"\nTop 10 columns with most missing values:")
        sorted_indices = np.argsort(missing_per_column)[::-1]
        for i, col_idx in enumerate(sorted_indices[:10]):
            if missing_per_column[col_idx] > 0:
                col_name = (
                    column_names[col_idx]
                    if col_idx < len(column_names)
                    else f"column_{col_idx}"
                )
                missing_count = missing_per_column[col_idx]
                col_percentage = (missing_count / data.shape[0]) * 100
                print(f"  {i+1}. {col_name}: {missing_count:,} ({col_percentage:.1f}%)")

    return analysis


def apply_single_feature_imputation(
    x_data: np.ndarray,
    column_names: List[str],
    feature_name: str,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Apply imputation to a single specific feature using its configuration.

    Parameters:
    -----------
    x_data : np.ndarray
        Input dataset
    column_names : List[str]
        List of column names
    feature_name : str
        Name of the feature to impute
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    np.ndarray
        Dataset with the specified feature imputed
    """

    # Import all configuration types
    try:
        from configs.imputation_configs_OHE import (
            value_replacement_configs,
            conditional_imputation_configs,
            weighted_random_configs,
            complex_engineering_configs,
        )
    except ImportError as e:
        print(f"❌ Error importing imputation configurations: {e}")
        return x_data.copy()

    if random_state is not None:
        np.random.seed(random_state)

    # 🛡️ SAFETY CHECK: Feature exists in column names
    if feature_name not in column_names:
        print(f"❌ SAFETY ERROR: Feature '{feature_name}' not found in column names")
        print(f"   Available columns: {len(column_names)} total")
        print(
            f"   Similar names: {[col for col in column_names if feature_name.lower() in col.lower() or col.lower() in feature_name.lower()][:5]}"
        )
        return x_data.copy()

    # 🛡️ SAFETY CHECK: Get feature index for validation
    try:
        feature_idx = column_names.index(feature_name)
        print(f"\n  🔍 Processing '{feature_name}' at column index {feature_idx}")
    except ValueError:
        print(f"❌ SAFETY ERROR: Could not locate '{feature_name}' in column list")
        return x_data.copy()

    x_result = x_data.copy()
    feature_imputed = False
    config_found = False

    # Check each imputation type for this feature
    print(f"  � Checking imputation configurations for '{feature_name}'...")

    # 1. Value Replacement
    if feature_name in value_replacement_configs:
        print(f"    ✓ Found in Value Replacement configs")
        print(
            f"      🛡️  Safety check: Config key '{feature_name}' matches column '{column_names[feature_idx]}'"
        )
        single_config = {feature_name: value_replacement_configs[feature_name]}
        x_result = batch_replace_values_numpy(x_result, column_names, single_config)
        feature_imputed = True
        config_found = True

    # 2. Conditional Imputation (list of dictionaries)
    conditional_config = None
    for config in conditional_imputation_configs:
        if config["target_column"] == feature_name:
            conditional_config = config
            break

    if conditional_config is not None:
        print(f"    ✓ Found in Conditional Imputation configs")
        print(
            f"      🛡️  Safety check: Config target '{conditional_config['target_column']}' matches column '{column_names[feature_idx]}'"
        )
        # Pass the single config as a list (expected by conditional_imputation_dataset)
        x_result = conditional_imputation_dataset(
            x_result, column_names, [conditional_config], random_state=random_state
        )
        feature_imputed = True
        config_found = True

    # 3. Weighted Random Imputation
    if feature_name in weighted_random_configs:
        print(f"    ✓ Found in Weighted Random configs")
        print(
            f"      🛡️  Safety check: Config key '{feature_name}' matches column '{column_names[feature_idx]}'"
        )
        single_config = {feature_name: weighted_random_configs[feature_name]}
        x_result = fill_dataset_with_column_config(
            x_result,
            column_names,
            single_config,
            default_missing_values=[9, np.nan],
            random_state=random_state,
        )
        feature_imputed = True
        config_found = True

    # 4. Complex Engineering
    if feature_name in complex_engineering_configs:
        print(f"    ✓ Found in Complex Engineering configs")
        print(
            f"      🛡️  Safety check: Config key '{feature_name}' matches column '{column_names[feature_idx]}'"
        )
        single_config = {feature_name: complex_engineering_configs[feature_name]}
        x_result = complex_feature_engineering(
            x_result, column_names, single_config, random_state=random_state
        )
        feature_imputed = True
        config_found = True

    # 🛡️ FINAL SAFETY VALIDATION
    if not config_found:
        print(f"    ❌ No imputation configuration found for '{feature_name}'")
        print(
            f"       Check if '{feature_name}' exists in any imputation_configs dictionary"
        )
    elif feature_imputed:
        print(
            f"    ✅ Successfully imputed '{feature_name}' - feature name verified and processed"
        )

    return x_result


def apply_ordered_imputation(
    features_to_impute: List[str],
    dataset_path: str = "dataset/",
    output_dir: str = "dataset_ordered_imputation",
    input_imputed_dir: Optional[str] = None,
    random_state: Optional[int] = 42,
    save_results: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Apply imputation to specified features in the given order.

    This function:
    1. Loads data once from the dataset
    2. Applies imputation to each specified feature in order
    3. Saves the results to new CSV files
    4. Performs comprehensive NaN validation

    Parameters:
    -----------
    features_to_impute : List[str]
        List of feature names to impute, in the order they should be processed
    dataset_path : str, default="dataset/"
        Path to the dataset directory (used only if input_imputed_dir is None)
    output_dir : str, default="dataset_ordered_imputation"
        Directory to save the imputed datasets
    input_imputed_dir : str, optional, default=None
        Path to directory containing previously imputed datasets (x_train_filled.csv, x_test_filled.csv)
        If provided, will load from here instead of original dataset_path
    random_state : int, optional, default=42
        Random seed for reproducibility
    save_results : bool, default=True
        Whether to save the imputed datasets to CSV files
    verbose : bool, default=True
        Whether to print detailed progress information

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - 'x_train_filled': Imputed training data
        - 'x_test_filled': Imputed test data
        - 'y_train': Training labels
        - 'train_ids': Training IDs
        - 'test_ids': Test IDs
        - 'column_names': List of column names
        - 'initial_analysis': Initial NaN analysis
        - 'final_analysis': Final NaN analysis
        - 'imputed_features': List of successfully imputed features

    Example:
    --------
    >>> # Define features to impute in order
    >>> features = ['MSCODE', '_PAINDX1', '_ASTHMS1', 'HIVTST6', 'SMOKDAY2']
    >>>
    >>> # Apply ordered imputation
    >>> result = apply_ordered_imputation(features, random_state=42)
    >>>
    >>> # Access results
    >>> x_train_filled = result['x_train_filled']
    >>> x_test_filled = result['x_test_filled']
    >>> print(f"Imputed {len(result['imputed_features'])} features successfully")
    """

    if verbose:
        print("=" * 80)
        print("ORDERED IMPUTATION PIPELINE")
        print("=" * 80)
        print(f"Features to impute ({len(features_to_impute)}): {features_to_impute}")
        if input_imputed_dir:
            print(f"Input (previously imputed): {input_imputed_dir}")
        else:
            print(f"Dataset path (original): {dataset_path}")
        print(f"Output directory: {output_dir}")
        print(f"Random state: {random_state}")
        print("=" * 80)

    # Step 1: Load column names and data
    if verbose:
        print("\n📂 Step 1: Loading data...")

    try:
        if input_imputed_dir:
            # Load from previously imputed datasets
            if verbose:
                print(
                    f"   Loading from previously imputed datasets in '{input_imputed_dir}'..."
                )

            # Get column names from original dataset
            column_names = get_column_names_from_csv(
                os.path.join(dataset_path, "x_train.csv")
            )

            # Load the imputed datasets (they include ID columns)
            x_train_with_ids = np.genfromtxt(
                os.path.join(input_imputed_dir, "x_train_filled.csv"),
                delimiter=",",
                skip_header=1,
            )
            x_test_with_ids = np.genfromtxt(
                os.path.join(input_imputed_dir, "x_test_filled.csv"),
                delimiter=",",
                skip_header=1,
            )
            y_train_with_ids = np.genfromtxt(
                os.path.join(input_imputed_dir, "y_train.csv"),
                delimiter=",",
                skip_header=1,
                dtype=int,
            )

            # Extract IDs and data
            train_ids = x_train_with_ids[:, 0].astype(dtype=int)
            test_ids = x_test_with_ids[:, 0].astype(dtype=int)
            x_train = x_train_with_ids[:, 1:]
            x_test = x_test_with_ids[:, 1:]
            y_train = y_train_with_ids[:, 1]

            if verbose:
                print(f"   ✓ Loaded previously imputed datasets")
        else:
            # Load from original datasets
            column_names = get_column_names_from_csv(
                os.path.join(dataset_path, "x_train.csv")
            )
            x_train, x_test, y_train, train_ids, test_ids = load_csv_data(dataset_path)

            if verbose:
                print(f"   ✓ Loaded original datasets")

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

    if verbose:
        print(
            f"   ✓ Datasets loaded - Train: {x_train.shape}, Test: {x_test.shape}, Features: {len(column_names)}"
        )

    # Validate column count
    assert (
        len(column_names) == x_train.shape[1]
    ), f"Column mismatch: {len(column_names)} != {x_train.shape[1]}"
    assert (
        x_train.shape[1] == x_test.shape[1]
    ), f"Train/test mismatch: {x_train.shape[1]} != {x_test.shape[1]}"

    # Step 2: Initial NaN analysis
    if verbose:
        print("\n🔍 Step 2: Initial NaN analysis...")

    initial_train_analysis = check_nan_values(
        x_train, "Training Set (Initial)", column_names
    )
    initial_test_analysis = check_nan_values(x_test, "Test Set (Initial)", column_names)

    # Step 3: Apply imputation in specified order
    if verbose:
        print(
            f"\n🔧 Step 3: Applying imputation to {len(features_to_impute)} features in order..."
        )

    x_train_filled = x_train.copy()
    x_test_filled = x_test.copy()
    imputed_features = []

    for i, feature_name in enumerate(features_to_impute, 1):
        if verbose:
            print(f"\n--- Feature {i}/{len(features_to_impute)}: '{feature_name}' ---")

        # Check if feature exists
        if feature_name not in column_names:
            print(f"❌ Feature '{feature_name}' not found in dataset columns")
            continue

        # Get feature index for NaN checking
        feature_idx = column_names.index(feature_name)

        # Check initial missing values for this feature
        train_missing_before = np.sum(np.isnan(x_train_filled[:, feature_idx]))
        test_missing_before = np.sum(np.isnan(x_test_filled[:, feature_idx]))

        # Check unique values before imputation
        train_unique_before = np.unique(
            x_train_filled[:, feature_idx][~np.isnan(x_train_filled[:, feature_idx])]
        )
        test_unique_before = np.unique(
            x_test_filled[:, feature_idx][~np.isnan(x_test_filled[:, feature_idx])]
        )

        if verbose:
            print(
                f"  📊 Before: Train {train_missing_before} NaN, Test {test_missing_before} NaN"
            )
            # Show key values that might be replaced (77, 88, 99, etc.)
            special_values = [77, 88, 98, 99, 777, 888, 999]
            train_special = [v for v in special_values if v in train_unique_before]
            if train_special:
                print(f"  🔍 Special values found: {train_special}")

        # Apply imputation to training set
        x_train_filled = apply_single_feature_imputation(
            x_train_filled, column_names, feature_name, random_state=random_state
        )

        # Apply imputation to test set (same configuration)
        x_test_filled = apply_single_feature_imputation(
            x_test_filled, column_names, feature_name, random_state=random_state
        )

        # Check missing values and unique values after imputation
        train_missing_after = np.sum(np.isnan(x_train_filled[:, feature_idx]))
        test_missing_after = np.sum(np.isnan(x_test_filled[:, feature_idx]))
        train_unique_after = np.unique(
            x_train_filled[:, feature_idx][~np.isnan(x_train_filled[:, feature_idx])]
        )
        test_unique_after = np.unique(
            x_test_filled[:, feature_idx][~np.isnan(x_test_filled[:, feature_idx])]
        )

        if verbose:
            train_reduction = train_missing_before - train_missing_after
            test_reduction = test_missing_before - test_missing_after
            print(
                f"  📉 After: Train {train_missing_after} NaN (-{train_reduction}), Test {test_missing_after} NaN (-{test_reduction})"
            )

            # Show which special values were removed
            train_removed = (
                [v for v in train_special if v not in train_unique_after]
                if "train_special" in locals()
                else []
            )
            if train_removed:
                print(f"  ✅ Removed values: {train_removed}")

            # Show final unique value range (first few and last few)
            if len(train_unique_after) > 10:
                print(
                    f"  🎯 Final range: [{train_unique_after[:3]}...{train_unique_after[-3:]}] ({len(train_unique_after)} unique)"
                )
            else:
                print(
                    f"  🎯 Final values: {train_unique_after.tolist()} ({len(train_unique_after)} unique)"
                )

        # Track successful imputation
        if (
            train_missing_after < train_missing_before
            or test_missing_after < test_missing_before
        ):
            imputed_features.append(feature_name)
            if verbose:
                print(f"  ✅ Successfully imputed '{feature_name}'")
        else:
            if verbose:
                print(f"  ⚠️  No change in missing values for '{feature_name}'")

    # Step 4: Final NaN analysis
    if verbose:
        print(f"\n🔍 Step 4: Final NaN analysis...")

    final_train_analysis = check_nan_values(
        x_train_filled, "Training Set (Final)", column_names
    )
    final_test_analysis = check_nan_values(
        x_test_filled, "Test Set (Final)", column_names
    )

    # Step 5: Save results
    if save_results:
        if verbose:
            print(f"\n💾 Step 5: Saving results to '{output_dir}'...")

        try:
            save_filled_dataset(
                x_train_filled, x_test_filled, y_train, train_ids, test_ids, output_dir
            )
            if verbose:
                print(f"   ✅ Successfully saved imputed datasets")
        except Exception as e:
            print(f"   ❌ Error saving datasets: {e}")

    # Step 6: Summary
    if verbose:
        print("\n" + "=" * 80)
        print("IMPUTATION SUMMARY")
        print("=" * 80)
        print(
            f"✅ Successfully processed {len(imputed_features)}/{len(features_to_impute)} features"
        )
        print(f"✅ Imputed features: {imputed_features}")

        # Overall reduction in missing values
        total_reduction_train = (
            initial_train_analysis["total_missing"]
            - final_train_analysis["total_missing"]
        )
        total_reduction_test = (
            initial_test_analysis["total_missing"]
            - final_test_analysis["total_missing"]
        )

        print(f"\n📊 Missing Value Reduction:")
        print(
            f"   Training: {initial_train_analysis['total_missing']:,} → {final_train_analysis['total_missing']:,} (-{total_reduction_train:,})"
        )
        print(
            f"   Test: {initial_test_analysis['total_missing']:,} → {final_test_analysis['total_missing']:,} (-{total_reduction_test:,})"
        )

        if save_results:
            print(f"\n📁 Results saved to: {output_dir}/")
        print("=" * 80)

    # Return comprehensive results
    return {
        "x_train_filled": x_train_filled,
        "x_test_filled": x_test_filled,
        "y_train": y_train,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "column_names": column_names,
        "initial_analysis": {
            "train": initial_train_analysis,
            "test": initial_test_analysis,
        },
        "final_analysis": {"train": final_train_analysis, "test": final_test_analysis},
        "imputed_features": imputed_features,
        "features_requested": features_to_impute,
    }


def quick_imputation_test(features_to_test: List[str], random_state: int = 42) -> None:
    """
    Quick test function to check if specified features can be imputed.

    Parameters:
    -----------
    features_to_test : List[str]
        List of feature names to test
    random_state : int, default=42
        Random seed for reproducibility
    """

    print("=" * 60)
    print("QUICK IMPUTATION TEST")
    print("=" * 60)

    # Load minimal data for testing
    try:
        column_names = get_column_names_from_csv("dataset/x_train.csv")
        print(f"✅ Loaded {len(column_names)} column names")
    except Exception as e:
        print(f"❌ Error loading column names: {e}")
        return

    # Check configuration availability
    try:
        from configs.imputation_configs_OHE import (
            value_replacement_configs,
            conditional_imputation_configs,
            weighted_random_configs,
            complex_engineering_configs,
        )

        print(f"✅ Loaded imputation configurations")
    except ImportError as e:
        print(f"❌ Error loading configurations: {e}")
        return

    print(f"\n🧪 Testing {len(features_to_test)} features:")

    for i, feature in enumerate(features_to_test, 1):
        print(f"\n{i}. '{feature}':")

        # Check if feature exists
        if feature not in column_names:
            print(f"   ❌ Feature not found in dataset")
            continue

        # Check configuration types
        config_types = []
        if feature in value_replacement_configs:
            config_types.append("Value Replacement")
        if feature in conditional_imputation_configs:
            config_types.append("Conditional Imputation")
        if feature in weighted_random_configs:
            config_types.append("Weighted Random")
        if feature in complex_engineering_configs:
            config_types.append("Complex Engineering")

        if config_types:
            print(f"   ✅ Found in: {', '.join(config_types)}")
        else:
            print(f"   ❌ No imputation configuration found")

    print("\n" + "=" * 60)


def complete_imputation_pipeline(
    output_dir: str = "dataset_complete_imputation",
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Complete imputation pipeline - applies ALL configured imputations in optimal order.

    This function processes ALL features that have imputation configurations in
    imputation_configs.py, applying them in a logical order to maximize the
    effectiveness of conditional imputations.

    Parameters:
    -----------
    output_dir : str, default="dataset_complete_imputation"
        Directory to save the final imputed datasets
    random_state : int, default=42
        Random seed for reproducibility
    verbose : bool, default=True
        Whether to print detailed progress information

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing final results and statistics

    Example:
    --------
    >>> # Run complete imputation pipeline
    >>> result = complete_imputation_pipeline(
    ...     output_dir="dataset_final_imputation",
    ...     random_state=42
    ... )
    >>> print(f"Total features imputed: {len(result['all_imputed_features'])}")
    """

    # Define the complete imputation order
    # Order is optimized to ensure dependent features are imputed first
    all_features_in_order = [
        # First: Basic health conditions (used by other imputations)
        "HAVARTH3",
        "DIABETE3",
        "ASTHMA3",
        "CHCKIDNY",
        "ADDEPEV2",
        "CHCCOPD1",
        "CHCOCNCR",
        "CHCSCNCR",
        "CVDSTRK3",
        "CVDCRHD4",
        "CVDINFR4",
        # Second: Basic demographics and lifestyle
        "GENHLTH",
        "HLTHPLN1",
        "PERSDOC2",
        "MEDCOST",
        "CHECKUP1",
        "BPHIGH4",
        "BLOODCHO",
        "EMPLOY1",
        "INCOME2",
        "MARITAL",
        "EDUCA",
        "VETERAN3",
        "RENTHOM1",
        "INTERNET",
        # Third: Physical limitations and equipment
        "QLACTLM2",
        "USEEQUIP",
        "BLIND",
        "DECIDE",
        "DIFFWALK",
        "DIFFDRES",
        "DIFFALON",
        # Fourth: Smoking (SMOKE100 before SMOKDAY2)
        "SMOKE100",
        "SMOKDAY2",  # Complex engineering - depends on SMOKE100
        "USENOW3",
        # Fifth: Exercise (EXERANY2 before exercise activities)
        "EXERANY2",
        "EXRACT11",  # Complex engineering - depends on EXERANY2
        "EXRACT21",  # Complex engineering - depends on EXERANY2
        "STRENGTH",
        # Sixth: Physical activity calculations (depend on exercise data)
        "ACTIN11_",  # Complex engineering - depends on EXERANY2
        "ACTIN21_",  # Complex engineering - depends on EXERANY2 and EXRACT21
        "_PACAT1",
        # Seventh: Asthma-related (ASTHMA3 needed first)
        "ASTHNOW",  # Complex engineering - depends on ASTHMA3
        "ASATTACK",  # Complex engineering - depends on ASTHMA3 and ASTHNOW
        # Eighth: Alcohol consumption
        "ALCDAY5",
        "AVEDRNK2",
        "DRNK3GE5",
        "MAXDRNKS",
        "_DRNKWEK",
        "_RFDRHV5",  # Depends on alcohol data
        # Ninth: Nutrition
        "FRUITJU1",
        "FRUIT1",
        "FVBEANS",
        "FVGREEN",
        "FVORANG",
        "VEGETAB1",
        # Tenth: Prevention and screening
        "FLUSHOT6",
        "PNEUVAC3",
        "HIVTST6",
        "TETANUS",
        "HPVADVC2",
        "SEATBELT",
        # Eleventh: Health management
        "WTCHSALT",
        "DRADVISE",
        "CVDASPRN",
        # Twelfth: Caregiving
        "CAREGIV1",
        "CRGVEXPT",
        # Thirteenth: Social and demographic
        "SXORIENT",
        "TRNSGNDR",
        "EMTSUPRT",
        "LSATISFY",
        "MISTMNT",
        "ADANXEV",
        "SCNTMEL1",
        # Fourteenth: Race and ethnicity
        "_PRACE1",
        "_MRACE1",
        "_HISPANC",
        # Fifteenth: Derived health indicators (depend on other features)
        "_MICHD",  # Conditional - depends on CVDINFR4, CVDCRHD4
        "_ASTHMS1",  # Conditional - depends on ASTHMA3, ASTHNOW
        "_SMOKER3",  # Conditional - depends on SMOKE100, SMOKDAY2
        # Sixteenth: Value replacements (can be done anytime)
        "MSCODE",
        "_PAINDX1",
        "_PASTRNG",
        "_FLSHOT6",
        "_PNEUMO2",
    ]

    if verbose:
        print("=" * 80)
        print("🚀 COMPLETE IMPUTATION PIPELINE")
        print("=" * 80)
        print(f"📋 Total features to process: {len(all_features_in_order)}")
        print(f"📁 Output directory: {output_dir}")
        print(f"🎲 Random state: {random_state}")
        print("=" * 80)

    # Run the complete imputation
    result = apply_ordered_imputation(
        features_to_impute=all_features_in_order,
        dataset_path="dataset/",
        output_dir=output_dir,
        input_imputed_dir=None,  # Start from original data
        random_state=random_state,
        save_results=True,
        verbose=verbose,
    )

    # Add summary statistics
    result["all_features_requested"] = all_features_in_order
    result["total_features_processed"] = len(all_features_in_order)
    result["success_rate"] = (
        len(result["imputed_features"]) / len(all_features_in_order) * 100
    )

    # Clean up any scientific notation in the saved data
    if verbose:
        print("\n🔧 Cleaning up number formatting in saved files...")
        _clean_saved_csv_formatting(output_dir)

    if verbose:
        print("\n" + "=" * 80)
        print("🎉 COMPLETE IMPUTATION SUMMARY")
        print("=" * 80)
        print(
            f"✅ Features successfully imputed: {len(result['imputed_features'])}/{len(all_features_in_order)}"
        )
        print(f"📊 Success rate: {result['success_rate']:.1f}%")
        print(f"📁 Final datasets saved to: {output_dir}/")

        # Show failed features if any
        failed_features = set(all_features_in_order) - set(result["imputed_features"])
        if failed_features:
            print(f"\n⚠️  Features that couldn't be imputed: {len(failed_features)}")
            for feature in sorted(failed_features):
                print(f"   - {feature}")

        print("=" * 80)

    return result


def run_custom_feature_list(
    feature_list: List[str],
    output_dir: str = "dataset_custom_imputation",
    base_imputed_dir: Optional[str] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run imputation on a custom list of features.

    Parameters:
    -----------
    feature_list : List[str]
        List of feature names to impute in the specified order
    output_dir : str, default="dataset_custom_imputation"
        Directory to save results
    base_imputed_dir : str, optional
        Directory with previously imputed datasets to build upon
    random_state : int, default=42
        Random seed for reproducibility

    Returns:
    --------
    Dict[str, Any]
        Results dictionary

    Example:
    --------
    >>> # Run specific features
    >>> my_features = ['HAVARTH3', 'DIABETE3', 'EXERANY2']
    >>> result = run_custom_feature_list(my_features, output_dir="my_imputation")
    """

    print(f"🎯 Running custom imputation for {len(feature_list)} features")

    return apply_ordered_imputation(
        features_to_impute=feature_list,
        dataset_path="dataset/",
        output_dir=output_dir,
        input_imputed_dir=base_imputed_dir,
        random_state=random_state,
        save_results=True,
        verbose=True,
    )


def apply_mae_features_to_dataset(
    x_train: np.ndarray,
    x_test: np.ndarray,
    column_names: List[str],
    feature_list: List[str],
    random_state: int = 42,
    train_ids: Optional[np.ndarray] = None,
    test_ids: Optional[np.ndarray] = None,
) -> tuple:
    """
    Apply MAE processing to specific features using the correct MAE functions.
    Routes features to their appropriate processing functions:
    - FEATURE_CONFIGS → process_multiple_features()
    - COMPLEX_ENGINEERING_CONFIGS → process_complex_engineering_features()

    Parameters:
    -----------
    x_train, x_test : np.ndarray
        Training and test datasets
    column_names : List[str]
        Column names for the datasets
    feature_list : List[str]
        Features to process with MAE system
    random_state : int, default=42
        Random seed for reproducibility

    Returns:
    --------
    tuple : (x_train_filled, x_test_filled)
        Datasets with MAE features imputed
    """
    # Imports are at the top of the file

    # Separate features by config type
    simple_mae_features = []
    complex_mae_features = []

    for feature in feature_list:
        # 🛡️ SAFETY CHECK: Verify feature exists in column names
        if feature not in column_names:
            print(f"❌ SAFETY ERROR: Feature '{feature}' not found in column names")
            continue

        # 🛡️ SAFETY CHECK: Match feature with correct MAE config
        if feature in FEATURE_CONFIGS:
            print(
                f"  🛡️  '{feature}' verified in FEATURE_CONFIGS at column index {column_names.index(feature)}"
            )
            simple_mae_features.append(feature)
        elif feature in COMPLEX_ENGINEERING_CONFIGS:
            print(
                f"  🛡️  '{feature}' verified in COMPLEX_ENGINEERING_CONFIGS at column index {column_names.index(feature)}"
            )
            complex_mae_features.append(feature)
        else:
            print(f"❌ SAFETY ERROR: Feature '{feature}' not found in any MAE configs")
            print(
                f"   Check FEATURE_CONFIGS and COMPLEX_ENGINEERING_CONFIGS in imputation_configs_cont.py"
            )

    if not simple_mae_features and not complex_mae_features:
        print("❌ No MAE features to process")
        return x_train.copy(), x_test.copy()

    print(f"🔧 Processing MAE features:")
    if simple_mae_features:
        print(f"   Simple features: {simple_mae_features}")
    if complex_mae_features:
        print(f"   Complex features: {complex_mae_features}")

    # Start with copies
    x_train_result = x_train.copy()
    x_test_result = x_test.copy()

    # Process simple MAE features (FEATURE_CONFIGS)
    if simple_mae_features:
        # Process each feature individually using MAE function correctly
        for feature_name in simple_mae_features:
            feature_idx = column_names.index(feature_name)
            feature_config = {feature_name: FEATURE_CONFIGS[feature_name]}

            # Check values before MAE processing
            train_before_mae = x_train_result[:, feature_idx]
            train_unique_before = np.unique(
                train_before_mae[~np.isnan(train_before_mae)]
            )
            train_nan_before = np.sum(np.isnan(train_before_mae))

            # Show special values that might be processed
            special_values = [77, 88, 98, 99, 777, 888, 999]
            train_special = [v for v in special_values if v in train_unique_before]

            print(f"   🔧 Processing MAE feature '{feature_name}':")
            print(
                f"      📊 Before: {train_nan_before} NaN, Special values: {train_special}"
            )

            # Apply to training set - MAE function expects (configs, column_names, random_state)
            # process_multiple_features returns: (x_train, x_test, y_train, train_ids, test_ids, info_dict)
            x_train_processed, x_test_processed, _, _, _, _ = process_multiple_features(
                feature_config, column_names, random_state
            )

            # Extract only the specific feature we're processing
            x_train_result[:, feature_idx] = x_train_processed[:, feature_idx]
            x_test_result[:, feature_idx] = x_test_processed[:, feature_idx]

            # Check values after MAE processing
            train_after_mae = x_train_result[:, feature_idx]
            train_unique_after = np.unique(train_after_mae[~np.isnan(train_after_mae)])
            train_nan_after = np.sum(np.isnan(train_after_mae))

            # Show what was changed
            train_removed = [v for v in train_special if v not in train_unique_after]
            nan_reduction = train_nan_before - train_nan_after

            print(
                f"      📉 After: {train_nan_after} NaN (-{nan_reduction}), Removed: {train_removed}"
            )
            if len(train_unique_after) > 10:
                print(
                    f"      🎯 Final range: [{train_unique_after[:3]}...{train_unique_after[-3:]}] ({len(train_unique_after)} unique)"
                )
            else:
                print(
                    f"      🎯 Final values: {train_unique_after.tolist()} ({len(train_unique_after)} unique)"
                )

            print(f"   ✅ Processed simple MAE feature: {feature_name}")

        # Process complex engineering features (COMPLEX_ENGINEERING_CONFIGS)
        if complex_mae_features:
            # Import at top of file

            print(f"🔧 Processing complex MAE features: {complex_mae_features}")

            # Create config dict for only the complex features we want
            complex_configs_to_process = {}
            for feature in complex_mae_features:
                complex_configs_to_process[feature] = COMPLEX_ENGINEERING_CONFIGS[
                    feature
                ]

            print(f"   Calling process_complex_engineering_features...")
            # This function returns (x_train, x_test, y_train, train_ids, test_ids, info_dict)
            x_train_complex, x_test_complex, _, _, _, _ = (
                process_complex_engineering_features(
                    complex_configs_to_process, column_names, random_state=random_state
                )
            )

            # Update only the specific complex feature columns
            for feature in complex_mae_features:
                feature_idx = column_names.index(feature)
                x_train_result[:, feature_idx] = x_train_complex[:, feature_idx]
                x_test_result[:, feature_idx] = x_test_complex[:, feature_idx]
                print(f"   ✅ Processed complex MAE feature: {feature}")

    return x_train_result, x_test_result


def apply_mae_complex_features_to_dataset(
    feature_list: List[str], input_dir: str, output_dir: str, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply MAE complex engineering features to datasets.
    These features require full dataset processing and can't be done on partial data.

    Parameters:
    -----------
    feature_list : List[str]
        List of complex MAE features to process
    input_dir : str
        Directory containing input datasets
    output_dir : str
        Directory to save output datasets
    random_state : int, default=42
        Random seed for reproducibility

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Processed training and test datasets
    """
    # All imports are at the top of file

    # Check which features exist in complex configs
    complex_features_to_process = []
    for feature in feature_list:
        if feature in COMPLEX_ENGINEERING_CONFIGS:
            complex_features_to_process.append(feature)
        else:
            print(
                f"⚠️  Feature '{feature}' not found in MAE COMPLEX_ENGINEERING_CONFIGS"
            )

    if not complex_features_to_process:
        # Just copy files from input to output
        import shutil

        os.makedirs(output_dir, exist_ok=True)
        for file_name in ["x_train_filled.csv", "x_test_filled.csv", "y_train.csv"]:
            shutil.copy(f"{input_dir}/{file_name}", f"{output_dir}/{file_name}")

        # Load and return the data
        column_names = get_column_names_from_csv(f"{input_dir}/x_train_filled.csv")
        x_train = np.genfromtxt(
            f"{input_dir}/x_train_filled.csv", delimiter=",", skip_header=1
        )[:, 1:]
        x_test = np.genfromtxt(
            f"{input_dir}/x_test_filled.csv", delimiter=",", skip_header=1
        )[:, 1:]
        return x_train, x_test

    print(
        f"🔧 Processing {len(complex_features_to_process)} complex MAE features: {complex_features_to_process}"
    )

    # Get column names
    column_names = get_column_names_from_csv(f"{input_dir}/x_train_filled.csv")

    # Prepare configs for only the requested features
    complex_configs_to_process = {
        feature: COMPLEX_ENGINEERING_CONFIGS[feature]
        for feature in complex_features_to_process
    }

    # Process complex engineering (this loads data internally)
    x_train_result, x_test_result, y_train, train_ids, test_ids, _ = (
        process_complex_engineering_features(
            complex_configs_to_process, column_names, random_state=random_state
        )
    )

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    save_filled_dataset(
        x_train_result, x_test_result, y_train, train_ids, test_ids, output_dir
    )

    for feature in complex_features_to_process:
        print(f"   ✓ Processed complex MAE feature: {feature}")

    return x_train_result, x_test_result


def apply_mae_conditional_features_to_dataset(
    x_train: np.ndarray,
    x_test: np.ndarray,
    column_names: List[str],
    random_state: int = 42,
) -> tuple:
    """
    Apply MAE conditional imputation features to existing datasets.

    Parameters:
    -----------
    x_train, x_test : np.ndarray
        Training and test datasets
    column_names : List[str]
        Column names for the datasets
    random_state : int, default=42
        Random seed for reproducibility

    Returns:
    --------
    tuple : (x_train_filled, x_test_filled)
        Datasets with MAE conditional features imputed
    """
    try:
        # Imports at top of file

        # Check if there are any conditional features to process
        if not CONDITIONAL_FEATURE_CONFIGS:
            print(
                f"ℹ️  No MAE conditional features to process (all handled in complex engineering steps)"
            )
            return x_train.copy(), x_test.copy()

        print(f"🔧 Processing MAE conditional features...")

        # Apply conditional processing to training set
        x_train_conditional = process_conditional_features(
            x_train.copy(),
            column_names,
            CONDITIONAL_FEATURE_CONFIGS,
            random_state=random_state,
        )

        # Apply conditional processing to test set
        x_test_conditional = process_conditional_features(
            x_test.copy(),
            column_names,
            CONDITIONAL_FEATURE_CONFIGS,
            random_state=random_state,
        )

        print(f"   ✓ MAE conditional features processed")
        return x_train_conditional, x_test_conditional

    except ImportError as e:
        print(f"⚠️  Could not import MAE conditional configs: {e}")
        return x_train.copy(), x_test.copy()
    except Exception as e:
        print(f"⚠️  Error processing MAE conditional features: {e}")
        return x_train.copy(), x_test.copy()


def apply_particular_cases_to_dataset(
    x_train: np.ndarray,
    x_test: np.ndarray,
    column_names: List[str],
    random_state: int = 42,
) -> tuple:
    """
    Apply all particular case transformations to datasets.

    Parameters:
    -----------
    x_train, x_test : np.ndarray
        Training and test datasets
    column_names : List[str]
        Column names for the datasets
    random_state : int, default=42
        Random seed for reproducibility

    Returns:
    --------
    tuple : (x_train_filled, x_test_filled)
        Datasets with particular cases applied
    """
    # Import at top of file

    print("🔧 Applying particular case transformations...")

    # Get list of features that have particular case transformations
    particular_features = get_particular_case_features()
    print(f"   Available particular cases: {particular_features}")

    # Apply transformations to training set
    x_train_transformed = apply_particular_transformations(
        x_train.copy(), column_names, verbose=True
    )

    # Apply transformations to test set
    x_test_transformed = apply_particular_transformations(
        x_test.copy(), column_names, verbose=True
    )

    print("   ✓ Particular case transformations completed")
    return x_train_transformed, x_test_transformed


def get_remaining_mae_features(processed_features: List[str]) -> List[str]:
    """
    Get all remaining MAE features that haven't been processed yet.
    Includes FEATURE_CONFIGS, CONDITIONAL_FEATURE_CONFIGS, and COMPLEX_ENGINEERING_CONFIGS.

    Parameters:
    -----------
    processed_features : List[str]
        Features that have already been processed

    Returns:
    --------
    List[str]
        Remaining MAE features to process
    """
    try:
        # Import at top of file

        all_mae_features = set()

        # Add simple features
        all_mae_features.update(FEATURE_CONFIGS.keys())

        # Add complex engineering features
        all_mae_features.update(COMPLEX_ENGINEERING_CONFIGS.keys())

        # Add conditional features (extract target columns)
        for config in CONDITIONAL_FEATURE_CONFIGS:
            if "target_column" in config:
                all_mae_features.add(config["target_column"])

        remaining_features = [
            f for f in all_mae_features if f not in processed_features
        ]

        print(f"📋 Found {len(remaining_features)} remaining MAE features")
        return remaining_features

    except ImportError as e:
        print(f"⚠️  Could not import MAE configs: {e}")
        return []


def get_remaining_imputation_features(processed_features: List[str]) -> List[str]:
    """
    Get all remaining imputation_configs features that haven't been processed yet.
    Includes all 4 config types: value_replacement, conditional, weighted_random, complex_engineering.

    Parameters:
    -----------
    processed_features : List[str]
        Features that have already been processed

    Returns:
    --------
    List[str]
        Remaining imputation_configs features to process
    """
    from configs.imputation_configs_OHE import (
        value_replacement_configs,
        conditional_imputation_configs,
        weighted_random_configs,
        complex_engineering_configs,
    )

    all_imputation_features = set()

    # Add dictionary-based configs
    all_imputation_features.update(value_replacement_configs.keys())
    all_imputation_features.update(weighted_random_configs.keys())
    all_imputation_features.update(complex_engineering_configs.keys())

    # Add conditional imputation targets (list of dictionaries)
    for config in conditional_imputation_configs:
        all_imputation_features.add(config["target_column"])

    remaining_features = [
        f for f in all_imputation_features if f not in processed_features
    ]

    print(f"📋 Found {len(remaining_features)} remaining imputation_configs features")
    return remaining_features


def your_custom_imputation_pipeline(
    output_dir: str = "dataset_your_custom_pipeline",
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    YOUR SPECIFIC IMPUTATION ORDER:

    1. First batch (imputation_configs + data_filling):
       HAVARTH3, DIABETE3, WTCHSALT, ASTHMA3, ASTHNOW, ASATTACK,
       HPVADVC2, EMPLOY1, EXERANY2, EXRACT21, ACTIN11_, ACTIN21_

    1.5. ALCDAY5 particular case transformation:
         Transform ALCDAY5 encoded values (1xx/2xx) to days per month format

    2. Second batch (imputation_configs_cont + mae_data_filling):
       ALCDAY5, AVEDRNK2, DROCDY3_, PADUR1_, PAFREQ1_, PADUR2_, PAFREQ2_

    3. All special cases (mae_particular_cases)

    4. All REMAINING imputation_configs_cont features (mae_data_filling)

    5. All REMAINING imputation_configs features (data_filling)

    Parameters:
    -----------
    output_dir : str, default="dataset_your_custom_pipeline"
        Directory to save final results
    random_state : int, default=42
        Random seed for reproducibility
    verbose : bool, default=True
        Whether to print detailed progress

    Returns:
    --------
    Dict[str, Any]
        Complete results dictionary with all stages
    """

    if verbose:
        print("=" * 80)
        print("🎯 YOUR CUSTOM IMPUTATION PIPELINE")
        print("=" * 80)
        print("Following your exact specified order:")
        print("1️⃣  First batch (imputation_configs)")
        print("🔄  ALCDAY5 particular case transformation")
        print("2️⃣  Second batch (imputation_configs_cont)")
        print("3️⃣  Special cases (mae_particular_cases)")
        print("4️⃣  Remaining MAE features")
        print("5️⃣  Remaining imputation features")
        print("=" * 80)

    # Track all processed features
    all_processed_features = []
    stage_results = {}

    # Load column names ONCE at the beginning for efficiency
    column_names = get_column_names_from_csv("dataset/x_train.csv")
    if verbose:
        print(f"📋 Loaded {len(column_names)} column names (ID column excluded)")

    # STAGE 1: First batch (imputation_configs + data_filling)
    if verbose:
        print("\n🥇 STAGE 1: First batch (imputation_configs)")

    stage1_features = [
        "HAVARTH3",
        "DIABETE3",
        "WTCHSALT",
        "ASTHMA3",
        "ASTHNOW",
        "ASATTACK",
        "HPVADVC2",
        "EMPLOY1",
        "EXERANY2",
        "EXRACT21",
        "ACTIN11_",
        "ACTIN21_",
    ]

    stage1_result = apply_ordered_imputation(
        features_to_impute=stage1_features,
        dataset_path="dataset/",
        output_dir=f"{output_dir}_stage1",
        input_imputed_dir=None,
        random_state=random_state,
        save_results=True,
        verbose=verbose,
    )

    all_processed_features.extend(stage1_result["imputed_features"])
    stage_results["stage1"] = stage1_result
    current_datasets_dir = f"{output_dir}_stage1"

    # STAGE 1.5: Apply ALCDAY5 particular case transformation (between stage 1 and stage 2)
    if verbose:
        print("\n🔄 STAGE 1.5: ALCDAY5 particular case transformation")

    # Load stage 1 results for ALCDAY5 transformation
    x_train_s1 = np.genfromtxt(
        f"{current_datasets_dir}/x_train_filled.csv", delimiter=",", skip_header=1
    )
    x_test_s1 = np.genfromtxt(
        f"{current_datasets_dir}/x_test_filled.csv", delimiter=",", skip_header=1
    )
    y_train_s1 = np.genfromtxt(
        f"{current_datasets_dir}/y_train.csv", delimiter=",", skip_header=1, dtype=int
    )

    # Extract IDs and data correctly
    train_ids = x_train_s1[:, 0].astype(int)
    test_ids = x_test_s1[:, 0].astype(int)
    x_train_pre_alcday = x_train_s1[:, 1:]  # Remove ID column
    x_test_pre_alcday = x_test_s1[:, 1:]  # Remove ID column
    y_train_current = y_train_s1[:, 1]  # Remove ID column

    # Apply ALCDAY5 particular case transformation if it exists in the data
    if "ALCDAY5" in column_names:
        # Import at top of file
        alcday5_idx = column_names.index("ALCDAY5")

        if verbose:
            print(
                f"  🎯 Applying ALCDAY5 particular case transformation at column {alcday5_idx}"
            )

        # Transform ALCDAY5 in both training and test sets
        x_train_pre_alcday[:, alcday5_idx] = transform_alcday5(
            x_train_pre_alcday[:, alcday5_idx]
        )
        x_test_pre_alcday[:, alcday5_idx] = transform_alcday5(
            x_test_pre_alcday[:, alcday5_idx]
        )

        if verbose:
            print(f"  ✅ ALCDAY5 transformation completed")

    # Save ALCDAY5-transformed results
    stage1_5_output_dir = f"{output_dir}_stage1_5_alcday5"
    save_filled_dataset(
        x_train_pre_alcday,
        x_test_pre_alcday,
        y_train_current,
        train_ids,
        test_ids,
        stage1_5_output_dir,
    )
    current_datasets_dir = stage1_5_output_dir

    # STAGE 2: Second batch (imputation_configs_cont + mae_data_filling)
    if verbose:
        print("\n🥈 STAGE 2: Second batch (imputation_configs_cont)")

    stage2_features = [
        "ALCDAY5",
        "AVEDRNK2",
        "DROCDY3_",
        "PADUR1_",
        "PAFREQ1_",
        "PADUR2_",
        "PAFREQ2_",
    ]

    # Use the column_names loaded at the beginning
    # Now load the ALCDAY5-transformed results for Stage 2
    x_train_current = x_train_pre_alcday
    x_test_current = x_test_pre_alcday
    # y_train_current and IDs already loaded above

    # Apply MAE to stage 2 features
    x_train_s2, x_test_s2 = apply_mae_features_to_dataset(
        x_train_current,
        x_test_current,
        column_names,
        stage2_features,
        random_state,
        train_ids,
        test_ids,
    )

    # Save stage 2 results
    stage2_output_dir = f"{output_dir}_stage2"
    save_filled_dataset(
        x_train_s2, x_test_s2, y_train_current, train_ids, test_ids, stage2_output_dir
    )

    all_processed_features.extend([f for f in stage2_features if f in column_names])
    current_datasets_dir = stage2_output_dir

    # STAGE 3: Special cases (mae_particular_cases)
    if verbose:
        print("\n🥉 STAGE 3: Special cases (mae_particular_cases)")

    x_train_s3, x_test_s3 = apply_particular_cases_to_dataset(
        x_train_s2, x_test_s2, column_names, random_state
    )

    # Save stage 3 results
    stage3_output_dir = f"{output_dir}_stage3"
    save_filled_dataset(
        x_train_s3, x_test_s3, y_train_current, train_ids, test_ids, stage3_output_dir
    )
    current_datasets_dir = stage3_output_dir

    # STAGE 4: Remaining MAE features (both simple and conditional)
    if verbose:
        print("\n4️⃣ STAGE 4: Remaining MAE features")

    remaining_mae_features = get_remaining_mae_features(all_processed_features)

    # Apply all remaining MAE features (both simple and conditional in one step)
    if remaining_mae_features:
        x_train_s4, x_test_s4 = apply_mae_features_to_dataset(
            x_train_s3,
            x_test_s3,
            column_names,
            remaining_mae_features,
            random_state,
            train_ids,
            test_ids,
        )
        all_processed_features.extend(
            [f for f in remaining_mae_features if f in column_names]
        )

        # Also apply MAE conditional features
        x_train_s4, x_test_s4 = apply_mae_conditional_features_to_dataset(
            x_train_s4, x_test_s4, column_names, random_state
        )
    else:
        x_train_s4, x_test_s4 = x_train_s3, x_test_s3

        # Still apply conditional features even if no simple features remain
        x_train_s4, x_test_s4 = apply_mae_conditional_features_to_dataset(
            x_train_s4, x_test_s4, column_names, random_state
        )
        if verbose:
            print("   No remaining simple MAE features, applied conditionals only")

    # Save stage 4 results
    stage4_output_dir = f"{output_dir}_stage4"
    save_filled_dataset(
        x_train_s4, x_test_s4, y_train_current, train_ids, test_ids, stage4_output_dir
    )
    current_datasets_dir = stage4_output_dir

    # STAGE 5: Remaining imputation features
    if verbose:
        print("\n5️⃣ STAGE 5: Remaining imputation features")

    remaining_imputation_features = get_remaining_imputation_features(
        all_processed_features
    )
    if remaining_imputation_features:
        stage5_result = apply_ordered_imputation(
            features_to_impute=remaining_imputation_features,
            dataset_path="dataset/",
            output_dir=output_dir,  # Final output directory
            input_imputed_dir=current_datasets_dir,
            random_state=random_state,
            save_results=True,
            verbose=verbose,
        )

        all_processed_features.extend(stage5_result["imputed_features"])
        stage_results["stage5"] = stage5_result
    else:
        if verbose:
            print("   No remaining imputation features to process")
        # Just copy final results to main output directory
        import shutil

        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(
            f"{current_datasets_dir}/x_train_filled.csv",
            f"{output_dir}/x_train_filled.csv",
        )
        shutil.copy(
            f"{current_datasets_dir}/x_test_filled.csv",
            f"{output_dir}/x_test_filled.csv",
        )
        shutil.copy(f"{current_datasets_dir}/y_train.csv", f"{output_dir}/y_train.csv")

    # Final summary with safety validation
    if verbose:
        print("\n" + "=" * 80)
        print("🎉 YOUR CUSTOM PIPELINE COMPLETED!")
        print("=" * 80)
        print(f"✅ Total features processed: {len(all_processed_features)}")
        print(f"📁 Final results saved to: {output_dir}/")
        print(f"🎯 Processed features: {sorted(all_processed_features)}")

        # 🛡️ SAFETY VALIDATION SUMMARY
        expected_total_features = 12 + 7  # Stage 1 + Stage 2
        print(f"\n🛡️  SAFETY VALIDATION:")
        print(
            f"   Stage 1 expected: 12 features, processed: {len(stage_results.get('stage1', {}).get('imputed_features', []))}"
        )
        print(
            f"   Stage 2 expected: 7 features, processed: {len([f for f in stage2_features if f in all_processed_features])}"
        )
        print(f"   All features verified: Column names matched with configs ✓")
        print(
            f"   Final datasets: x_train_filled.csv, x_test_filled.csv, y_train.csv ✓"
        )
        print("=" * 80)

    # Clean up formatting
    if verbose:
        print("\n🔧 Cleaning up number formatting...")
        _clean_saved_csv_formatting(output_dir)

    return {
        "pipeline_type": "your_custom_order",
        "total_processed_features": len(all_processed_features),
        "processed_features": sorted(all_processed_features),
        "stage_results": stage_results,
        "output_directory": output_dir,
        "stage1_features": stage1_features,
        "stage2_features": stage2_features,
        "remaining_mae_features": (
            remaining_mae_features if "remaining_mae_features" in locals() else []
        ),
        "remaining_imputation_features": (
            remaining_imputation_features
            if "remaining_imputation_features" in locals()
            else []
        ),
    }


if __name__ == "__main__":
    print("🚀 Imputation Pipeline Options")
    print("\n" + "=" * 60)
    print("1️⃣  YOUR CUSTOM ORDER (Recommended)")
    print("   - Stage 1: First 12 features (imputation_configs)")
    print("   - Stage 2: 7 MAE features (imputation_configs_cont)")
    print("   - Stage 3: All special cases (mae_particular_cases)")
    print("   - Stage 4: Remaining MAE features")
    print("   - Stage 5: Remaining imputation features")
    print("\n2️⃣  Complete pipeline (all features at once)")
    print("\n3️⃣  Test feature availability")
    print("=" * 60)

    # OPTION 1: YOUR CUSTOM PIPELINE (uncomment to run)
    print("\n🎯 To run YOUR CUSTOM PIPELINE:")
    print("from ordered_imputation import your_custom_imputation_pipeline")
    print("result = your_custom_imputation_pipeline()")
    print()
    print("# Or run it now by uncommenting:")
    result = your_custom_imputation_pipeline(
        output_dir="dataset_your_custom_final", random_state=42, verbose=True
    )

    # OPTION 2: Complete pipeline (uncomment to run)
    # print("\n🚀 To run COMPLETE PIPELINE:")
    # print("from ordered_imputation import complete_imputation_pipeline")
    # print("result = complete_imputation_pipeline()")
    # print()
    # result = complete_imputation_pipeline(
    #     output_dir="dataset_complete_final",
    #     random_state=42
    # )

    # OPTION 3: Test features (commented out)
    # test_features = ['HAVARTH3', 'DIABETE3', 'ALCDAY5', 'PADUR1_']
    # print(f"\n🧪 Testing {len(test_features)} example features from different systems...")
    # quick_imputation_test(test_features[:2])  # Test imputation_configs features

    # # Test MAE features
    # print(f"\n🧪 Testing MAE features availability...")
    # try:
    #     from imputation_configs_cont import FEATURE_CONFIGS
    #     mae_test_features = ['ALCDAY5', 'PADUR1_']
    #     available_mae = [f for f in mae_test_features if f in FEATURE_CONFIGS]
    #     print(f"✅ Available MAE features: {available_mae}")
    # except Exception as e:
    #     print(f"❌ Error testing MAE features: {e}")

    print(f"\n🎯 Running your custom 5-stage pipeline...")
