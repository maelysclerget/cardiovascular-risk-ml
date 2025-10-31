"""
Script to remove specific features from the dataset.

This script uses the remove_features_from_filled module to remove a predefined list
of features from the dataset. Features are removed based on analysis of their
importance or correlation with other features.
"""

import os
from remove_features_from_filled import remove_features_and_save

features_to_exclude = [
    "feature_1",
    "feature_2",
    "feature_4",
    "feature_7",
    "feature_8",
    "feature_9",
    "feature_10",
    "feature_11",
    "feature_12",
    "feature_13",
    "feature_14",
    "feature_15",
    "feature_18",
    "feature_19",
    "feature_20",
    "feature_21",
    "feature_22",
    "feature_23",
    "feature_24",
    "feature_49",
    "feature_54",
    "feature_62",
    "feature_63",
    "feature_81",
    "feature_82",
    "feature_83",
    "feature_84",
    "feature_85",
    "feature_86",
    "feature_89",
    "feature_90",
    "feature_92",
    "feature_93",
    "feature_94",
    "feature_127",
    "feature_128",
    "feature_129",
    "feature_130",
    "feature_131",
    "feature_132",
    "feature_133",
    "feature_134",
    "feature_135",
    "feature_145",
    "feature_191",
    "feature_196",
    "feature_197",
    "feature_216",
    "feature_217",
    "feature_219",
    "feature_220",
    "feature_221",
    "feature_222",
    "feature_223",
    "feature_224",
    "feature_226",
    "feature_228",
    "feature_229",
    "feature_231",
    "feature_235",
    "feature_236",
    "feature_238",
    "feature_242",
    "feature_243",
    "feature_244",
    "feature_245",
    "feature_246",
    "feature_247",
    "feature_250",
    "feature_255",
    "feature_256",
    "feature_257",
    "feature_258",
    "feature_260",
    "feature_274",
    "feature_275",
    "feature_276",
    "feature_277",
    "feature_278",
    "feature_279",
    "feature_280",
    "feature_281",
    "feature_284",
    "feature_291",
    "feature_292",
    "feature_293",
    "feature_294",
    "feature_298",
    "feature_299",
    "feature_300",
    "feature_301",
    "feature_304",
    "feature_307",
    "feature_308",
    "feature_309",
    "feature_311",
    "feature_312",
    "feature_313",
    "feature_314",
    "feature_315",
    "feature_316",
    "feature_317",
    "feature_320",
]

input_dir = os.path.join(
    "data", "processed", "dataset_your_custom_pipeline", "dataset_your_custom_final"
)
output_dir = os.path.join("data", "processed", "dataset_features_removed")


def main():
    """
    Main function to execute the feature removal process.
    Removes specified features from the dataset and saves the results
    in a new directory with a summary of removed features.
    """
    print(f"Starting feature removal process...")
    print(f"Number of features to remove: {len(features_to_exclude)}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    remove_features_and_save(
        features_to_remove=features_to_exclude,
        input_dir=input_dir,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
