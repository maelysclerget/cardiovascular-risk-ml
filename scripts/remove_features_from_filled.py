import os
import numpy as np


def save_csv_preserving_format(
    input_path: str, output_path: str, columns_to_remove: set
):
    """
    Save a CSV file while preserving its original format and removing specified columns.

    Parameters:
    -----------
    input_path : str
        Path to the input CSV file
    output_path : str
        Path where the output CSV file will be saved
    columns_to_remove : set
        Set of column indices to remove (1-based indexing)
    """
    # Read header and data separately to preserve format
    with open(input_path, "r") as f:
        header = f.readline().strip()
        header_cols = header.split(",")

    # Read data with numpy
    data = np.genfromtxt(input_path, delimiter=",", skip_header=1)

    # Determine columns to keep (convert from 1-based to 0-based indexing)
    cols_to_keep = [i for i in range(data.shape[1]) if i + 1 not in columns_to_remove]

    # Filter columns
    filtered_data = data[:, cols_to_keep]
    filtered_header = [header_cols[i] for i in cols_to_keep]

    # Save with original format preserved
    with open(output_path, "w") as f:
        # Write header
        f.write(",".join(filtered_header) + "\n")
        # Write data with preserved format
        np.savetxt(f, filtered_data, delimiter=",", fmt="%.6f")


def remove_features_and_save(
    features_to_remove: list,
    input_dir: str = os.path.join(
        "data", "processed", "dataset_your_custom_pipeline", "dataset_your_custom_final"
    ),
    output_dir: str = os.path.join("data", "processed", "dataset_features_removed"),
):
    """
    Remove specified features from filled datasets and save to a new directory.

    Parameters:
    -----------
    features_to_remove : list
        List of feature names to remove (e.g., ['feature_1', 'feature_2', 'feature_4'])
    input_dir : str
        Directory containing filled datasets
    output_dir : str
        Directory to save the filtered datasets
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Convert feature names to column indices (add 1 for ID column)
    columns_to_remove = set()
    feature_mapping = {}

    for feature in features_to_remove:
        try:
            # Extract the index from feature_N format directly
            idx = int(feature.split("_")[1]) + 1  # +1 for ID column
            columns_to_remove.add(idx)
            feature_mapping[f"feature_{idx-1}"] = feature
            print(f"Will remove feature: {feature}")
        except (IndexError, ValueError):
            print(
                f"Warning: Invalid feature format {feature}, expected 'feature_N' format"
            )

    # Save the feature removal summary
    with open(os.path.join(output_dir, "feature_removal_summary.txt"), "w") as f:
        f.write("Removed features:\n")
        for idx in sorted(columns_to_remove):
            feature_name = f"feature_{idx-1}"
            f.write(f"{feature_name}\n")

    # Process each dataset file
    for filename in ["x_train_filled.csv", "x_test_filled.csv"]:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        save_csv_preserving_format(input_path, output_path, columns_to_remove)

    # Copy y_train.csv as is
    input_path = os.path.join(input_dir, "y_train.csv")
    output_path = os.path.join(output_dir, "y_train.csv")
    save_csv_preserving_format(input_path, output_path, set())

    print(f"\nProcessing complete!")
    print(f"Removed {len(columns_to_remove)} features")
    print(f"Results saved in: {output_dir}")
    print("See feature_removal_summary.txt for details of removed features")
