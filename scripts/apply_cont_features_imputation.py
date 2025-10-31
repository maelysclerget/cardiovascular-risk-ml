"""
Single-Continuous Feature Imputation Implementation

This script provides a function to impute ONE SINGLE FEATURE from the configs/imputation_configs_cont.py
configurations. It allows precise control over the imputation process by handling one
feature at a time, which is particularly useful for features requiring complex or
specialized imputation logic.

Purpose:
    - Impute a single feature from imputation_configs_cont.py configurations
    - Allow detailed monitoring and validation of the imputation process
    - Enable feature-specific customization of imputation steps

Example Implementation:
    The script includes an example implementation for feature_303 (PAVIG11_ - vigorous
    activity time) which demonstrates complex conditional imputation based on
    feature_290 (ACTIN11_). This serves as a template showing how to:
    1. Set up feature-specific configuration
    2. Handle dependent features (e.g., PAVIG11_ depends on ACTIN11_)
    3. Apply multi-step imputation logic

Dependencies:
    - src.cont_features_data_filling: Complex imputation functions
    - configs.imputation_configs_cont: Feature-specific configurations
    - numpy: Data handling

Note:
    While this script can process any feature from imputation_configs_cont.py, the main()
    function specifically demonstrates the imputation of feature_303 as an example.
    To process a different feature, modify the configuration and feature name in
    the main() function.
"""

import numpy as np
import os
from typing import List, Dict, Optional
from src.cont_features_data_filling import (
    complex_multi_step_preprocessing,
)  # Handles complex imputation steps
from configs.imputation_configs_cont import (
    FEATURE_CONFIGS,  # Simple feature configurations
    CONDITIONAL_FEATURE_CONFIGS,  # Configurations with conditional logic
    COMPLEX_ENGINEERING_CONFIGS,  # Multi-step imputation configs
)


def apply_imputation_for_feature(
    feature_name: str,
    config: List[Dict],
    input_dir: str = "HISTORICAL_PATH_EXAMPLE/dataset_your_custom_final_stage1",
    output_dir: str = "HISTORICAL_PATH_EXAMPLE/dataset_your_custom_final_stage1",
    random_state: Optional[int] = 42,
):
    """
    Reference implementation of feature-specific imputation.
    This function shows how individual features were processed using configurations
    from imputation_configs_cont.py during the data cleaning phase.

    Note: This is a historical reference implementation. The actual data processing
    has been completed and the filled datasets are available in their final form.

    Parameters:
    -----------
    feature_name : str
        Name of the feature that was imputed (e.g., 'feature_303' for PAVIG11_)
    config : List[Dict]
        Configuration defining the imputation steps that were applied
    input_dir : str
        Historical reference to input directory (kept for documentation)
    output_dir : str
        Historical reference to output directory (kept for documentation)
    random_state : int, optional
        Random seed used for reproducibility
    """
    print(f"\n{'='*80}")
    print(f"APPLYING IMPUTATION FOR {feature_name}")
    print(f"{'='*80}")

    # Load data
    print("\nLoading data...")
    x_train = np.genfromtxt(
        os.path.join(input_dir, "x_train_filled.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(input_dir, "x_test_filled.csv"), delimiter=",", skip_header=1
    )
    y_train = np.genfromtxt(
        os.path.join(input_dir, "y_train.csv"), delimiter=",", skip_header=1
    )

    # Get IDs and data
    train_ids = x_train[:, 0]
    test_ids = x_test[:, 0]
    x_train = x_train[:, 1:]  # Remove ID column
    x_test = x_test[:, 1:]  # Remove ID column

    # Create column names
    n_features = x_train.shape[1]
    column_names = [f"feature_{i}" for i in range(n_features)]

    # Show initial state
    feature_idx = column_names.index(feature_name)
    print(f"\nInitial state for {feature_name}:")
    print(f"Train shape: {x_train.shape}")
    print(f"Test shape: {x_test.shape}")
    print(f"Train unique values: {np.unique(x_train[:, feature_idx])}")
    print(f"Test unique values: {np.unique(x_test[:, feature_idx])}")
    print(f"Train NaN count: {np.sum(np.isnan(x_train[:, feature_idx]))}")
    print(f"Test NaN count: {np.sum(np.isnan(x_test[:, feature_idx]))}")

    # Apply imputation
    print("\nApplying imputation...")
    try:
        step_results = complex_multi_step_preprocessing(
            x_train=x_train,
            x_test=x_test,
            feature_name=feature_name,
            column_names=column_names,
            steps_config=config,
            random_state=random_state,
        )

        print("\nFinal state:")
        print(f"Train unique values: {np.unique(x_train[:, feature_idx])}")
        print(f"Test unique values: {np.unique(x_test[:, feature_idx])}")
        print(f"Train NaN count: {np.sum(np.isnan(x_train[:, feature_idx]))}")
        print(f"Test NaN count: {np.sum(np.isnan(x_test[:, feature_idx]))}")

        # Save results
        print(f"\nSaving results to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)

        # Add back the IDs as first column
        x_train_with_ids = np.column_stack((train_ids, x_train))
        x_test_with_ids = np.column_stack((test_ids, x_test))

        # Save to files
        header = "Id," + ",".join(column_names)
        np.savetxt(
            os.path.join(output_dir, "x_train_filled.csv"),
            x_train_with_ids,
            fmt="%.6g",
            delimiter=",",
            header=header,
            comments="",
        )

        np.savetxt(
            os.path.join(output_dir, "x_test_filled.csv"),
            x_test_with_ids,
            fmt="%.6g",
            delimiter=",",
            header=header,
            comments="",
        )

        print("\nImputation completed and files saved!")
        return step_results

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return None


def main():
    """
    Main function configured specifically for feature_303 (PAVIG11_ - vigorous activity time).

    This implementation demonstrates complex conditional imputation where:
    1. For people with vigorous activity (feature_290 == 2):
       - NaN values are redistributed within valid range (0-19200 minutes)
    2. For people without vigorous activity (feature_290 < 2):
       - NaN values are set to 0 (no activity time)
    3. For people with unknown activity status (feature_290 is NaN):
       - NaN values are set to 0 (conservative assumption)

    Other features can be processed by modifying the configuration and feature name.
    Commented code shows how to process all features from imputation_configs_cont.py.
    """
    print(f"\n{'='*80}")
    print("IMPUTATION APPLICATION - FEATURE 303 (PAVIG11_)")
    print(f"{'='*80}")

    # Available configurations (uncomment to process all features):
    # print("\nAvailable configurations:")
    # print(f"Simple features: {len(FEATURE_CONFIGS)}")
    # print(f"Conditional features: {len(CONDITIONAL_FEATURE_CONFIGS)}")
    # print(f"Complex engineering: {len(COMPLEX_ENGINEERING_CONFIGS)} features")
    #
    # Process all complex features:
    # for feature_name, config in COMPLEX_ENGINEERING_CONFIGS.items():
    #     print(f"\nProcessing {feature_name}...")
    #     apply_imputation_for_feature(feature_name, config)

    # Currently configured for PAVIG11_ (vigorous activity time)
    feature_name = "feature_303"  # PAVIG11_ (vigorous activity time)
    config = [
        {
            "step": "conditional_redistribute",
            "condition_column": "feature_290",  # ACTIN11_
            "condition_value": 2,  # Has vigorous activity
            "values_to_redistribute": [
                np.nan
            ],  # Replace NaN values for people with vigorous activity
            "valid_range": (0, 19200),  # Based on actual data range: 0-14400 minutes
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "feature_290",  # ACTIN11_
            "condition_value": lambda x: x < 2,  # No vigorous activity
            "target_values": [
                np.nan
            ],  # Replace NaN values for people with no vigorous activity
            "fill_value": 0,  # No vigorous activity time for people who don't do vigorous activities
        },
        {
            # Additional step: Handle any remaining NaN values
            # This could happen if ACTIN11_ itself has NaN values
            "step": "conditional_fixed_fill",
            "condition_column": "feature_290",  # ACTIN11_
            "condition_value": lambda x: np.isnan(x),  # NaN in ACTIN11_
            "target_values": [
                np.nan
            ],  # Replace NaN values where activity status is unknown
            "fill_value": 0,  # Assume no vigorous activity when activity status is unknown
        },
    ]

    # Example of how feature_303 was processed
    apply_imputation_for_feature(
        feature_name=feature_name,
        config=config,
        # Note: These paths are historical references
        input_dir="HISTORICAL_PATH_EXAMPLE/dataset_your_custom_final_stage1",
        output_dir="HISTORICAL_PATH_EXAMPLE/dataset_your_custom_final_stage1",
    )


if __name__ == "__main__":
    main()
