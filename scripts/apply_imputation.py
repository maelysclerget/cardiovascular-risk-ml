#!/usr/bin/env python3
"""
Simple imputation script for both train and test datasets.

This script applies all imputation configurations from configs/imputation_configs_OHE.py
to both training and test datasets using the exact same process. It uses multi-step
imputation including value replacements, conditional fills, and statistical methods.

Dependencies:
    - src.helpers: Data loading utilities
    - src.data_filling: Imputation functions and configuration handlers
    - configs.imputation_configs_OHE: Imputation rules and configurations

Usage:
    python apply_imputation.py

Input files (from data/raw/):
    - x_train.csv: Training features
    - x_test.csv: Test features
    - y_train.csv: Training labels

Output files (to data/processed/dataset_your_custom_pipeline/):
    - x_train_filled.csv: Imputed training features
    - x_test_filled.csv: Imputed test features
    - y_train.csv: Unchanged training labels

The script will:
1. Load the original datasets from data/raw/
2. Extract column names (ensuring correct indexing)
3. Apply configured imputation methods to training set
4. Apply same imputation configuration to test set
5. Save the filled datasets to CSV files
"""

import numpy as np
from src.helpers import load_csv_data  # Handles data loading and basic preprocessing
from data_filling import (
    get_column_names_from_csv,  # Ensures consistent column indexing
    apply_imputation_configurations,  # Applies multi-step imputation
    save_filled_dataset,  # Handles proper CSV export
)


def main():
    """
    Main function to apply imputation to both train and test datasets.
    """
    print("=== IMPUTATION PIPELINE ===")
    print("Applying all imputation configurations to train and test datasets...")

    # Step 1: Get column names from CSV (ensures correct indexing)
    print("\n1. Extracting column names...")
    column_names = get_column_names_from_csv("data/raw/x_train.csv")

    # Step 2: Load original datasets
    print("\n2. Loading datasets...")
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data/raw/")

    print(f"   Training set: {x_train.shape}")
    print(f"   Test set: {x_test.shape}")
    print(f"   Column names: {len(column_names)}")

    # Verify column count matches
    assert (
        len(column_names) == x_train.shape[1]
    ), f"Column mismatch: {len(column_names)} != {x_train.shape[1]}"
    assert (
        x_train.shape[1] == x_test.shape[1]
    ), f"Train/test column mismatch: {x_train.shape[1]} != {x_test.shape[1]}"

    # Check initial missing values
    initial_train_missing = np.sum(np.isnan(x_train))
    initial_test_missing = np.sum(np.isnan(x_test))

    print(
        f"   Initial missing values - Train: {initial_train_missing:,}, Test: {initial_test_missing:,}"
    )

    # Step 3: Apply imputation to TRAINING SET
    print(f"\n3. Applying imputation to TRAINING SET...")
    print("-" * 60)
    x_train_filled = apply_imputation_configurations(
        x_train, column_names, random_state=42
    )

    # Step 4: Apply imputation to TEST SET (same process)
    print(f"\n4. Applying imputation to TEST SET...")
    print("-" * 60)
    x_test_filled = apply_imputation_configurations(
        x_test, column_names, random_state=42  # Same seed for consistency
    )

    # Step 5: Verify results
    print(f"\n5. Verification...")
    final_train_missing = np.sum(np.isnan(x_train_filled))
    final_test_missing = np.sum(np.isnan(x_test_filled))

    print(
        f"   Final missing values - Train: {final_train_missing:,}, Test: {final_test_missing:,}"
    )
    print(
        f"   Reduction - Train: {initial_train_missing - final_train_missing:,}, Test: {initial_test_missing - final_test_missing:,}"
    )

    # Step 6: Save filled datasets
    print(f"\n6. Saving filled datasets...")
    output_dir = "data/processed/dataset_your_custom_pipeline"
    save_filled_dataset(
        x_train_filled, x_test_filled, y_train, train_ids, test_ids, output_dir
    )

    print(f"\n=== IMPUTATION COMPLETED SUCCESSFULLY ===")
    print(f"✓ Training set: {x_train.shape} → {x_train_filled.shape}")
    print(f"✓ Test set: {x_test.shape} → {x_test_filled.shape}")
    print(f"✓ Datasets saved to '{output_dir}/'")
    print(f"✓ Column indexing verified and correct")

    return x_train_filled, x_test_filled


if __name__ == "__main__":
    main()
