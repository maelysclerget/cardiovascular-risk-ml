"""
MAE Data Imputation Application System
=====================================

This module provides the main application system for continuous feature data filling.
It implements a comprehensive pipeline for handling missing values and special codes in
continuous/ordinal variables using various imputation strategies.

The system supports:
1. Particular case transformations (custom logic for specific features)
2. Simple feature preprocessing (basic replacements and redistributions)
3. Conditional feature preprocessing (context-aware transformations)
4. Complex multi-step engineering (sophisticated feature engineering)

Key Features:
- Handles both training and test sets consistently
- Supports multiple imputation strategies
- Provides detailed processing statistics
- Maintains data integrity and distribution

Example:
    >>> from mae_testing import apply_mae_imputation_configurations
    >>> from helpers import load_csv_data
    >>> from data_filling import get_column_names_from_csv
    >>>
    >>> # Load data
    >>> column_names = get_column_names_from_csv('dataset/x_train.csv')
    >>> x_train, x_test, y_train, train_ids, test_ids = load_csv_data('dataset/')
    >>>
    >>> # Apply MAE imputation
    >>> x_train_filled, info = apply_mae_imputation_configurations(
    ...     x_train, column_names, random_state=42
    ... )
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import sys
import os

# Add current directory to path for imports
sys.path.append(".")


def apply_mae_imputation_configurations(
    x_data: np.ndarray,
    column_names: List[str],
    random_state: Optional[int] = None,
    include_simple: bool = True,
    include_conditional: bool = True,
    include_complex: bool = True,
    include_particular: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply all MAE imputation configurations to a dataset in the correct order.
    This is the MAE equivalent of apply_imputation_configurations() from data_filling.py.

    Order of operations:
    1. Particular Cases (mae_particular_cases.py transformations)
    2. Simple Feature Configs (FEATURE_CONFIGS from mae_dictionnary.py)
    3. Conditional Feature Configs (CONDITIONAL_FEATURE_CONFIGS)
    4. Complex Multi-Step Engineering (COMPLEX_ENGINEERING_CONFIGS)

    Parameters:
    -----------
    x_data : np.ndarray
        Input dataset (2D array: rows=samples, columns=features)
    column_names : List[str]
        List of column names corresponding to features (must match x_data.shape[1])
    random_state : int, optional
        Random seed for reproducibility
    include_simple : bool, default=True
        Whether to apply simple feature preprocessing from FEATURE_CONFIGS
    include_conditional : bool, default=True
        Whether to apply conditional redistribution from CONDITIONAL_FEATURE_CONFIGS
    include_complex : bool, default=True
        Whether to apply complex engineering from COMPLEX_ENGINEERING_CONFIGS
    include_particular : bool, default=True
        Whether to apply particular cases transformations

    Returns:
    --------
    Tuple[np.ndarray, Dict[str, Any]]
        - Dataset with all MAE imputations applied
        - Dictionary with processing information and statistics

    Example:
    --------
    >>> # Get column names
    >>> from data_filling import get_column_names_from_csv
    >>> column_names = get_column_names_from_csv('dataset/x_train.csv')
    >>>
    >>> # Load data
    >>> from helpers import load_csv_data
    >>> x_train, x_test, y_train, train_ids, test_ids = load_csv_data('dataset/')
    >>>
    >>> # Apply MAE imputation to training set
    >>> x_train_filled, train_info = apply_mae_imputation_configurations(
    ...     x_train, column_names, random_state=42
    ... )
    >>>
    >>> # Apply MAE imputation to test set
    >>> x_test_filled, test_info = apply_mae_imputation_configurations(
    ...     x_test, column_names, random_state=42
    ... )
    """

    print("=" * 80)
    print("MAE COMPREHENSIVE IMPUTATION PIPELINE")
    print("=" * 80)
    print(f"Dataset shape: {x_data.shape}")
    print(f"Number of features: {len(column_names)}")
    print(
        f"Processing order: {'Particular → ' if include_particular else ''}{'Simple → ' if include_simple else ''}{'Conditional → ' if include_conditional else ''}{'Complex' if include_complex else ''}"
    )
    print("=" * 80)

    # Import MAE configurations
    try:
        from configs.imputation_configs_cont import (
            FEATURE_CONFIGS,
            CONDITIONAL_FEATURE_CONFIGS,
            COMPLEX_ENGINEERING_CONFIGS,
        )

        print(f"✓ MAE configurations imported successfully")
        print(f"  Simple features: {len(FEATURE_CONFIGS)}")
        print(f"  Conditional features: {len(CONDITIONAL_FEATURE_CONFIGS)}")
        print(f"  Complex features: {len(COMPLEX_ENGINEERING_CONFIGS)}")
    except ImportError as e:
        print(f"❌ ERROR: Could not import MAE configurations: {e}")
        return x_data.copy(), {"error": f"Import error: {e}"}

    # Initialize processing info
    processing_info = {
        "steps_completed": [],
        "features_processed": {},
        "errors": [],
        "statistics": {},
    }

    # Create working copy
    x_processed = x_data.copy()

    if random_state is not None:
        np.random.seed(random_state)

    # =================================================================
    # STEP 1: PARTICULAR CASES TRANSFORMATIONS
    # =================================================================
    if include_particular:
        print(f"\n🔸 STEP 1: PARTICULAR CASES TRANSFORMATIONS")
        try:
            # Import mae_particular_cases functions
            from src.cont_features_particular_cases import (
                apply_particular_transformations,
            )

            # Apply particular cases (this handles multi-column dependencies)
            x_processed = apply_particular_transformations(
                x_processed.copy(), column_names, random_state=random_state
            )

            processing_info["steps_completed"].append("particular_cases")
            print(f"✓ Particular cases transformations completed")

        except ImportError:
            print(
                f"⚠️ Warning: mae_particular_cases not available, skipping particular transformations"
            )
        except Exception as e:
            error_msg = f"Error in particular cases: {e}"
            print(f"❌ {error_msg}")
            processing_info["errors"].append(error_msg)

    # =================================================================
    # STEP 2: SIMPLE FEATURE PREPROCESSING
    # =================================================================
    if include_simple and FEATURE_CONFIGS:
        print(f"\n🔸 STEP 2: SIMPLE FEATURE PREPROCESSING")
        try:
            from src.cont_features_data_filling import process_multiple_features

            x_processed, _, _, _, _, simple_info = process_multiple_features(
                FEATURE_CONFIGS, column_names, random_state=random_state
            )

            processing_info["steps_completed"].append("simple_features")
            processing_info["features_processed"].update(simple_info)
            print(
                f"✓ Simple feature preprocessing completed ({len(FEATURE_CONFIGS)} features)"
            )

        except Exception as e:
            error_msg = f"Error in simple features: {e}"
            print(f"❌ {error_msg}")
            processing_info["errors"].append(error_msg)

    # =================================================================
    # STEP 3: CONDITIONAL FEATURE PREPROCESSING
    # =================================================================
    if include_conditional and CONDITIONAL_FEATURE_CONFIGS:
        print(f"\n🔸 STEP 3: CONDITIONAL FEATURE PREPROCESSING")
        try:
            from src.cont_features_data_filling import process_conditional_features

            x_processed, _, _, _, _, conditional_info = process_conditional_features(
                CONDITIONAL_FEATURE_CONFIGS, column_names, random_state=random_state
            )

            processing_info["steps_completed"].append("conditional_features")
            processing_info["features_processed"].update(conditional_info)
            print(
                f"✓ Conditional feature preprocessing completed ({len(CONDITIONAL_FEATURE_CONFIGS)} features)"
            )

        except Exception as e:
            error_msg = f"Error in conditional features: {e}"
            print(f"❌ {error_msg}")
            processing_info["errors"].append(error_msg)

    # =================================================================
    # STEP 4: COMPLEX MULTI-STEP ENGINEERING
    # =================================================================
    if include_complex and COMPLEX_ENGINEERING_CONFIGS:
        print(f"\n🔸 STEP 4: COMPLEX MULTI-STEP ENGINEERING")
        try:
            from src.cont_features_data_filling import (
                process_complex_engineering_features,
            )

            x_processed, _, _, _, _, complex_info = (
                process_complex_engineering_features(
                    COMPLEX_ENGINEERING_CONFIGS, column_names, random_state=random_state
                )
            )

            processing_info["steps_completed"].append("complex_engineering")
            processing_info["features_processed"].update(complex_info)
            print(
                f"✓ Complex engineering completed ({len(COMPLEX_ENGINEERING_CONFIGS)} features)"
            )

        except Exception as e:
            error_msg = f"Error in complex engineering: {e}"
            print(f"❌ {error_msg}")
            processing_info["errors"].append(error_msg)

    # =================================================================
    # FINAL STATISTICS
    # =================================================================
    print(f"\n" + "=" * 80)
    print(f"MAE IMPUTATION PIPELINE COMPLETED")
    print(f"=" * 80)
    print(f"Steps completed: {len(processing_info['steps_completed'])}")
    print(f"Features processed: {len(processing_info['features_processed'])}")
    print(f"Errors encountered: {len(processing_info['errors'])}")

    if processing_info["errors"]:
        print(f"\nErrors:")
        for error in processing_info["errors"]:
            print(f"  ❌ {error}")

    # Calculate processing statistics
    nan_before = np.sum(np.isnan(x_data))
    nan_after = np.sum(np.isnan(x_processed))
    processing_info["statistics"] = {
        "nan_before": int(nan_before),
        "nan_after": int(nan_after),
        "nan_filled": int(nan_before - nan_after),
        "fill_percentage": (
            float((nan_before - nan_after) / nan_before * 100)
            if nan_before > 0
            else 0.0
        ),
    }

    print(f"\nFinal Statistics:")
    print(f"  NaN values before: {processing_info['statistics']['nan_before']:,}")
    print(f"  NaN values after: {processing_info['statistics']['nan_after']:,}")
    print(f"  NaN values filled: {processing_info['statistics']['nan_filled']:,}")
    print(f"  Fill percentage: {processing_info['statistics']['fill_percentage']:.2f}%")
    print("=" * 80)

    return x_processed, processing_info


if __name__ == "__main__":
    """Example usage of the MAE imputation pipeline."""
    from helpers import load_csv_data
    from data_filling import get_column_names_from_csv

    # Load data and column names
    column_names = get_column_names_from_csv("dataset/x_train.csv")
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data("dataset/")

    print("Applying MAE imputation pipeline...")

    # Process training data
    print("\nProcessing training data...")
    x_train_filled, train_info = apply_mae_imputation_configurations(
        x_train, column_names, random_state=42
    )

    # Process test data
    print("\nProcessing test data...")
    x_test_filled, test_info = apply_mae_imputation_configurations(
        x_test, column_names, random_state=42
    )

    # Show results summary
    print("\nResults Summary:")
    print("\nTraining Data:")
    print(f"- NaN values filled: {train_info['statistics']['nan_filled']:,}")
    print(f"- Fill percentage: {train_info['statistics']['fill_percentage']:.1f}%")
    print(f"- Steps completed: {', '.join(train_info['steps_completed'])}")

    print("\nTest Data:")
    print(f"- NaN values filled: {test_info['statistics']['nan_filled']:,}")
    print(f"- Fill percentage: {test_info['statistics']['fill_percentage']:.1f}%")
    print(f"- Steps completed: {', '.join(test_info['steps_completed'])}")

    print("=" * 80)
    print("COMPREHENSIVE MAE IMPUTATION TYPE TESTING")
    print("=" * 80)
    print("Testing one feature from each of the 4 MAE imputation types:")
    print("1. Particular Cases: 'HHADULT' (NaN → NUMADULT)")
    print("2. Simple Feature: 'PHYSHLTH' (88→0, redistribute 77,99,NaN)")
    print("3. Conditional Feature: (TBD - based on available configs)")
    print("4. Complex Engineering: 'JOINPAIN' (arthritis-conditional)")
    print("=" * 80)
