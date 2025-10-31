import numpy as np
from typing import Dict, List, Optional
import csv
import os


def read_csv_with_header(filename):
    """Read a CSV file with headers and return numpy array."""
    with open(filename, "r") as f:
        # Read header
        header = next(csv.reader(f))
        # Reset file pointer and skip header
        f.seek(0)
        # Load data with numpy, skip header row
        data = np.genfromtxt(f, delimiter=",", skip_header=1)
        if data.ndim == 1:
            # Handle single-column case
            data = data.reshape(-1, 1)
    return header, data


"""
MAE PARTICULAR CASES - CUSTOM TRANSFORMATIONS ONLY
=================================================

This file contains ONLY the custom transformation logic for features that need 
special encoding/decoding before they can be processed by the standard Mae system.

After transformation here, the features are processed normally in mae_dictionnary.py
"""

# =============================================================================
# CUSTOM TRANSFORMATION FUNCTIONS
# =============================================================================


def transform_alcday5(data):
    """
    Transform     # Map    # Map feature names to their numeric indices
    # _MINAC11 = feature_295 (original index: 295)
    # PADUR1_ = feature_291 (original index: 291)
    # PAFREQ1_ = feature_293 (original index: 293)
    required_cols = ['feature_295', 'feature_291', 'feature_293']
    feature_mapping = {
        '_MINAC11': 'feature_295',
        'PADUR1_': 'feature_291',
        'PAFREQ1_': 'feature_293'
    }ames to their numeric indices
    # _MINAC11 = feature_295
    # PADUR1_ = feature_293
    # PAFREQ1_ = feature_294
    required_cols = ['feature_295', 'feature_291', 'feature_293']
    feature_mapping = {
        '_MINAC11': 'feature_295',
        'PADUR1_': 'feature_291',
        'PAFREQ1_': 'feature_293'
    }coded values to days per month format.

    ONLY handles encoding conversion:
    - 1xx values: Days per week → (subtract 100, multiply by 4.28 for days/month)
    - 2xx values: Days per month → (subtract 200)
    - All other values (777, 888, 999, NaN): Left unchanged for mae_dictionnary.py

    Args:
        data (np.array): ALCDAY5 column data

    Returns:
        np.array: Transformed data with 1xx/2xx converted to days per month
    """
    transformed_data = data.copy()

    # Transform weekly values (101-107) to days per month
    weekly_mask = (transformed_data >= 101) & (transformed_data <= 199)
    if np.any(weekly_mask):
        # Convert: (value - 100) days/week × 4.34 weeks/month = days/month
        transformed_data[weekly_mask] = (transformed_data[weekly_mask] - 100) * 4.28

    # Transform monthly values (201-230) to days per month
    monthly_mask = (transformed_data >= 201) & (transformed_data <= 299)
    if np.any(monthly_mask):
        # Convert: value - 200 = days/month
        transformed_data[monthly_mask] = transformed_data[monthly_mask] - 200

    # Leave 777, 888, 999, NaN unchanged for Mae system to handle
    return transformed_data


def transform_drnkwek(drnkwek_data, avedrnk2_data, drocdy3_data):
    """
    Transform _DRNKWEK special codes using calculated weekly alcohol consumption.

    ONLY handles special code replacement:
    - 99900: Replace with AVEDRNK2 * DROCDY3_ * 7 (calculated weekly drinks)
    - All other values: Left unchanged for mae_dictionnary.py

    Args:
        drnkwek_data (np.array): _DRNKWEK column data
        avedrnk2_data (np.array): AVEDRNK2 column data (drinks per occasion)
        drocdy3_data (np.array): DROCDY3_ column data (days per week drinking)

    Returns:
        np.array: Transformed data with 99900 replaced by calculated values
    """
    transformed_data = drnkwek_data.copy()

    # Find rows where _DRNKWEK = 99900 (special code)
    special_code_mask = transformed_data == 99900

    if np.any(special_code_mask):
        # Calculate weekly consumption: drinks per occasion × days per week × 7
        # Only calculate for rows with the special code
        calculated_values = (
            avedrnk2_data[special_code_mask] * drocdy3_data[special_code_mask] * 7
        )

        # Handle cases where calculation inputs might be NaN or special codes
        # Only replace where calculation is valid (not NaN)
        valid_calculation = ~np.isnan(calculated_values)

        if np.any(valid_calculation):
            # Create a mask for rows that have special code AND valid calculation
            final_mask = np.zeros_like(transformed_data, dtype=bool)
            final_mask[special_code_mask] = valid_calculation

            # Replace special code with calculated values
            transformed_data[final_mask] = calculated_values[valid_calculation]

    # Leave all other values unchanged for Mae system to handle
    return transformed_data


def transform_minac11(minac11_data, padur1_data, pafreq1_data):
    """
    Transform _MINAC11 by replacing NaN values with calculated physical activity minutes.

    ONLY handles NaN replacement:
    - NaN values: Replace with PADUR1_ * PAFREQ1_ (duration × frequency = total minutes)
    - All other values: Left unchanged for mae_dictionnary.py

    Args:
        minac11_data (np.array): _MINAC11 column data (minutes of activity)
        padur1_data (np.array): PADUR1_ column data (duration per session)
        pafreq1_data (np.array): PAFREQ1_ column data (frequency per week)

    Returns:
        np.array: Transformed data with NaN replaced by calculated values
    """
    transformed_data = minac11_data.copy()

    # Find rows where _MINAC11 is NaN
    nan_mask = np.isnan(transformed_data)

    if np.any(nan_mask):
        # Calculate total minutes: duration per session × frequency per week
        calculated_values = padur1_data[nan_mask] * pafreq1_data[nan_mask]

        # Handle cases where calculation inputs might be NaN or special codes
        # Only replace where calculation is valid (not NaN)
        valid_calculation = ~np.isnan(calculated_values)

        if np.any(valid_calculation):
            # Create a mask for rows that have NaN AND valid calculation
            final_mask = np.zeros_like(transformed_data, dtype=bool)
            final_mask[nan_mask] = valid_calculation

            # Replace NaN with calculated values
            transformed_data[final_mask] = calculated_values[valid_calculation]

    # Leave all other values unchanged for Mae system to handle
    return transformed_data


def transform_minac21(minac21_data, padur2_data, pafreq2_data):
    """
    Transform _MINAC21 by replacing NaN values with calculated physical activity minutes.

    ONLY handles NaN replacement:
    - NaN values: Replace with PADUR2_ * PAFREQ2_ (duration × frequency = total minutes)
    - All other values: Left unchanged for mae_dictionnary.py

    Args:
        minac21_data (np.array): _MINAC21 column data (minutes of activity)
        padur2_data (np.array): PADUR2_ column data (duration per session)
        pafreq2_data (np.array): PAFREQ2_ column data (frequency per week)

    Returns:
        np.array: Transformed data with NaN replaced by calculated values
    """
    transformed_data = minac21_data.copy()

    # Find rows where _MINAC21 is NaN
    nan_mask = np.isnan(transformed_data)

    if np.any(nan_mask):
        # Calculate total minutes: duration per session × frequency per week
        calculated_values = padur2_data[nan_mask] * pafreq2_data[nan_mask]

        # Handle cases where calculation inputs might be NaN or special codes
        # Only replace where calculation is valid (not NaN)
        valid_calculation = ~np.isnan(calculated_values)

        if np.any(valid_calculation):
            # Create a mask for rows that have NaN AND valid calculation
            final_mask = np.zeros_like(transformed_data, dtype=bool)
            final_mask[nan_mask] = valid_calculation

            # Replace NaN with calculated values
            transformed_data[final_mask] = calculated_values[valid_calculation]

    # Leave all other values unchanged for Mae system to handle
    return transformed_data


def transform_longwtch(data):
    """
    Transform LONGWTCH encoded values to days format.

    ONLY handles encoding conversion:
    - 555: Treat as "don't know" → leave unchanged for Mae system distribution
    - 1xx values: Time per day → (subtract 100)
    - 2xx values: Time per week → (subtract 200, multiply by 7)
    - 3xx values: Time per month → (subtract 300, multiply by 30)
    - 4xx values: Time per year → (subtract 400, multiply by 365)
    - All other values (555, 777, 999, NaN): Left unchanged for mae_dictionnary.py

    Args:
        data (np.array): LONGWTCH column data

    Returns:
        np.array: Transformed data with encoded values converted to days
    """
    transformed_data = data.copy()

    # Leave 555 unchanged - treat as special "don't know" code for Mae system
    # This allows it to be handled by distribution alongside 777, 999

    # Transform daily values (1xx) - remove the 1__ prefix
    daily_mask = (transformed_data >= 100) & (transformed_data <= 199)
    if np.any(daily_mask):
        # Convert: value - 100 = days
        transformed_data[daily_mask] = transformed_data[daily_mask] - 100

    # Transform weekly values (2xx) - remove 2__ prefix and multiply by 7
    weekly_mask = (transformed_data >= 200) & (transformed_data <= 299)
    if np.any(weekly_mask):
        # Convert: (value - 200) weeks × 7 days/week = days
        transformed_data[weekly_mask] = (transformed_data[weekly_mask] - 200) * 7

    # Transform monthly values (3xx) - remove 3__ prefix and multiply by 30
    monthly_mask = (transformed_data >= 300) & (transformed_data <= 399)
    if np.any(monthly_mask):
        # Convert: (value - 300) months × 30 days/month = days
        transformed_data[monthly_mask] = (transformed_data[monthly_mask] - 300) * 30

    # Transform yearly values (4xx) - remove 4__ prefix and multiply by 365
    yearly_mask = (transformed_data >= 400) & (transformed_data <= 499)
    if np.any(yearly_mask):
        # Convert: (value - 400) years × 365 days/year = days
        transformed_data[yearly_mask] = (transformed_data[yearly_mask] - 400) * 365

    # Leave 777, 999, NaN unchanged for Mae system to handle
    return transformed_data


def transform_bldsugar(data):
    """
    Transform BLDSUGAR encoded values to per year format.

    ONLY handles encoding conversion:
    - 1xx values: Time per day → (subtract 100, multiply by 365 for per year)
    - 2xx values: Time per week → (subtract 200, multiply by 52 for per year)
    - 3xx values: Time per month → (subtract 300, multiply by 12 for per year)
    - 4xx values: Time per year → (subtract 400)
    - All other values (777, 888, 999, NaN): Left unchanged for mae_dictionnary.py

    Args:
        data (np.array): BLDSUGAR column data

    Returns:
        np.array: Transformed data with encoded values converted to per year
    """
    transformed_data = data.copy()

    # Transform daily values (1xx) - remove 1__ prefix and multiply by 365
    daily_mask = (transformed_data >= 100) & (transformed_data <= 199)
    if np.any(daily_mask):
        # Convert: (value - 100) times per day × 365 days/year = times per year
        transformed_data[daily_mask] = (transformed_data[daily_mask] - 100) * 365

    # Transform weekly values (2xx) - remove 2__ prefix and multiply by 52
    weekly_mask = (transformed_data >= 200) & (transformed_data <= 299)
    if np.any(weekly_mask):
        # Convert: (value - 200) times per week × 52 weeks/year = times per year
        transformed_data[weekly_mask] = (transformed_data[weekly_mask] - 200) * 52

    # Transform monthly values (3xx) - remove 3__ prefix and multiply by 12
    monthly_mask = (transformed_data >= 300) & (transformed_data <= 399)
    if np.any(monthly_mask):
        # Convert: (value - 300) times per month × 12 months/year = times per year
        transformed_data[monthly_mask] = (transformed_data[monthly_mask] - 300) * 12

    # Transform yearly values (4xx) - remove 4__ prefix
    yearly_mask = (transformed_data >= 400) & (transformed_data <= 499)
    if np.any(yearly_mask):
        # Convert: (value - 400) = times per year
        transformed_data[yearly_mask] = transformed_data[yearly_mask] - 400

    # Leave 777, 888, 999, NaN unchanged for Mae system to handle
    return transformed_data


def transform_feetchk2(data):
    """
    Transform FEETCHK2 encoded values to per year format.

    ONLY handles encoding conversion:
    - 1xx values: Time per day → (subtract 100, multiply by 365 for per year)
    - 2xx values: Time per week → (subtract 200, multiply by 52 for per year)
    - 3xx values: Time per month → (subtract 300, multiply by 12 for per year)
    - 4xx values: Time per year → (subtract 400)
    - All other values (777, 888, 999, NaN): Left unchanged for mae_dictionnary.py

    Args:
        data (np.array): FEETCHK2 column data

    Returns:
        np.array: Transformed data with encoded values converted to per year
    """
    transformed_data = data.copy()

    # Transform daily values (1xx) - remove 1__ prefix and multiply by 365
    daily_mask = (transformed_data >= 100) & (transformed_data <= 199)
    if np.any(daily_mask):
        # Convert: (value - 100) times per day × 365 days/year = times per year
        transformed_data[daily_mask] = (transformed_data[daily_mask] - 100) * 365

    # Transform weekly values (2xx) - remove 2__ prefix and multiply by 52
    weekly_mask = (transformed_data >= 200) & (transformed_data <= 299)
    if np.any(weekly_mask):
        # Convert: (value - 200) times per week × 52 weeks/year = times per year
        transformed_data[weekly_mask] = (transformed_data[weekly_mask] - 200) * 52

    # Transform monthly values (3xx) - remove 3__ prefix and multiply by 12
    monthly_mask = (transformed_data >= 300) & (transformed_data <= 399)
    if np.any(monthly_mask):
        # Convert: (value - 300) times per month × 12 months/year = times per year
        transformed_data[monthly_mask] = (transformed_data[monthly_mask] - 300) * 12

    # Transform yearly values (4xx) - remove 4__ prefix
    yearly_mask = (transformed_data >= 400) & (transformed_data <= 499)
    if np.any(yearly_mask):
        # Convert: (value - 400) = times per year
        transformed_data[yearly_mask] = transformed_data[yearly_mask] - 400

    # Leave 777, 888, 999, NaN unchanged for Mae system to handle
    return transformed_data


def transform_hivtstd3(data):
    """
    Transform HIVTSTD3 month/year values to categorical time periods.

    ONLY handles encoding conversion:
    - MMYYYY format: Extract year and categorize by time periods
    - 11985 (Jan 1985) → Category based on year 1985
    - 122016 (Dec 2016) → Category based on year 2016
    - Categories: 1=Very Old (1985-1995), 2=Old (1996-2005), 3=Recent (2006-2015), 4=Very Recent (2016+)
    - All other values (777, 888, 999, NaN): Left unchanged for mae_dictionnary.py

    Args:
        data (np.array): HIVTSTD3 column data

    Returns:
        np.array: Transformed data with month/year converted to categorical periods
    """
    transformed_data = data.copy()

    # Find values that look like month/year format (should be >= 11985 and <= 122025)
    # Valid range: month (1-12) + year (1985-2025) = 11985 to 122025
    month_year_mask = (transformed_data >= 11985) & (transformed_data <= 122025)

    if np.any(month_year_mask):
        month_year_values = transformed_data[month_year_mask]

        # Extract years from MMYYYY format
        # For values like 11985, 21985, ..., 121985: year = value % 10000 if value >= 101985
        # For values like 11985: year = 1985, month = 1
        years = np.zeros_like(month_year_values)

        # Handle different formats
        for i, val in enumerate(month_year_values):
            val_str = str(int(val))
            if len(val_str) == 5:  # Format: MYYYY (e.g., 11985)
                years[i] = int(val_str[1:])
            elif len(val_str) == 6:  # Format: MMYYYY (e.g., 122016)
                years[i] = int(val_str[2:])
            else:
                years[i] = val  # Keep as is if format is unclear

        # Categorize by time periods
        categories = np.zeros_like(years)
        categories[(years >= 1985) & (years <= 1995)] = 1  # Very Old
        categories[(years >= 1996) & (years <= 2005)] = 2  # Old
        categories[(years >= 2006) & (years <= 2015)] = 3  # Recent
        categories[years >= 2016] = 4  # Very Recent

        # Apply categories back to the original data
        transformed_data[month_year_mask] = categories

    # Leave 777, 888, 999, NaN unchanged for Mae system to handle
    return transformed_data


def transform_flshtmy2(data):
    """
    Transform FLSHTMY2 month/year values to categorical time periods.

    ONLY handles encoding conversion:
    - MMYYYY format: Categorize by specific date ranges (2 categories)
    - Category 1: 12014 to 062015 (Jan 2014 to Jun 2015) - Early period
    - Category 2: 072014+ onwards (Jul 2014 onwards, including all 2015-2016) - Later period
    - All other values (777777, 999999, NaN): Left unchanged for complex engineering

    Args:
        data (np.array): FLSHTMY2 column data

    Returns:
        np.array: Transformed data with month/year converted to categorical periods
    """
    transformed_data = data.copy()

    # Find values that are valid month/year format (actual range is 12014 to 122016)
    month_year_mask = (transformed_data >= 12014) & (transformed_data <= 122016)

    if np.any(month_year_mask):
        month_year_values = transformed_data[month_year_mask]

        # Categorize by specific date ranges - need to properly parse dates
        categories = np.zeros_like(month_year_values)

        for i, val in enumerate(month_year_values):
            val_int = int(val)
            val_str = str(val_int)

            # Parse month and year correctly
            if len(val_str) == 5:  # Format: MYYYY (e.g., 12014)
                month = int(val_str[0])
                year = int(val_str[1:])
            elif len(val_str) == 6:  # Format: MMYYYY (e.g., 122015)
                month = int(val_str[:2])
                year = int(val_str[2:])
            else:
                continue

            # Category 1: Jan 2014 to Jun 2015 (12014 to 062015)
            if (year == 2014) or (year == 2015 and month <= 6):
                categories[i] = 1
            # Category 2: Jul 2015 onwards (072015+, including all 2016)
            elif (year == 2015 and month > 6) or (year >= 2016):
                categories[i] = 2

        # Apply categories back to the original data
        transformed_data[month_year_mask] = categories

    # Leave 777777, 999999, NaN unchanged for complex engineering system to handle
    return transformed_data


def transform_hhadult(hhadult_data, numadult_data):
    """
    Transform HHADULT with conditional logic based on NUMADULT.

    ONLY handles NaN replacement:
    - For NaN: If NUMADULT has valid entry (not NaN), use that value
    - Leave 77, 99, and remaining NaN unchanged for other systems to handle

    Args:
        hhadult_data (np.array): HHADULT column data
        numadult_data (np.array): NUMADULT column data

    Returns:
        np.array: Transformed HHADULT data with NaN filled from NUMADULT where possible
    """
    transformed_data = hhadult_data.copy()

    # For NaN values, use NUMADULT if it has valid data (not NaN)
    nan_mask = np.isnan(transformed_data)
    numadult_valid_mask = ~np.isnan(numadult_data)

    # Fill NaN with NUMADULT where NUMADULT is valid (not NaN)
    can_fill_mask = nan_mask & numadult_valid_mask
    if np.any(can_fill_mask):
        transformed_data[can_fill_mask] = numadult_data[can_fill_mask]

    # Leave 77, 99, and remaining NaN unchanged for other systems to handle
    return transformed_data


# =============================================================================
# GENERAL TRANSFORMATION FUNCTION
# =============================================================================

# =============================================================================
# TRANSFORMATION REGISTRY
# =============================================================================

# Registry mapping feature names to their transformation functions
TRANSFORMATION_FUNCTIONS = {
    "ALCDAY5": transform_alcday5,
    "_DRNKWEK": transform_drnkwek,  # Special case: requires multiple columns
    "_MINAC11": transform_minac11,  # Special case: requires multiple columns
    "_MINAC21": transform_minac21,  # Special case: requires multiple columns
    "LONGWTCH": transform_longwtch,
    "BLDSUGAR": transform_bldsugar,
    "FEETCHK2": transform_feetchk2,
    "HIVTSTD3": transform_hivtstd3,
    "FLSHTMY2": transform_flshtmy2,
    "HHADULT": transform_hhadult,  # Special case: requires NUMADULT column
    # Add new transformations here as you create them:
    # 'OTHER_FEATURE': transform_other_feature,
    # 'SPECIAL_FEATURE': transform_special_feature,
}


def apply_particular_transformations(data, column_names, verbose=True):
    """
    Apply all particular case transformations to the dataset.

    This function automatically applies transformations for all registered
    particular case features. After this, use the regular Mae system
    for standard replacements and redistribution.

    Args:
        data (np.array): Full dataset
        column_names (list): Column names
        verbose (bool): Print transformation details

    Returns:
        np.array: Dataset with particular transformations applied
    """
    transformed_data = data.copy()
    transformations_applied = []

    if verbose:
        print("=== APPLYING PARTICULAR CASE TRANSFORMATIONS ===")

    # Apply transformations for all registered features that exist in data
    for feature_name, transform_func in TRANSFORMATION_FUNCTIONS.items():
        if feature_name in column_names:
            try:
                feature_idx = column_names.index(feature_name)
                if verbose:
                    print(f"  ✓ Transforming {feature_name}...")

                # Handle special multi-column transformations
                if feature_name == "_DRNKWEK":
                    # _DRNKWEK requires AVEDRNK2 and DROCDY3_ columns
                    required_cols = ["AVEDRNK2", "DROCDY3_"]
                    if all(col in column_names for col in required_cols):
                        avedrnk2_idx = column_names.index("AVEDRNK2")
                        drocdy3_idx = column_names.index("DROCDY3_")

                        original_data = transformed_data[:, feature_idx].copy()
                        transformed_data[:, feature_idx] = transform_func(
                            transformed_data[:, feature_idx],  # _DRNKWEK data
                            transformed_data[:, avedrnk2_idx],  # AVEDRNK2 data
                            transformed_data[:, drocdy3_idx],  # DROCDY3_ data
                        )
                    else:
                        missing_cols = [
                            col for col in required_cols if col not in column_names
                        ]
                        if verbose:
                            print(
                                f"    ✗ Missing required columns for {feature_name}: {missing_cols}"
                            )
                        continue
                elif feature_name == "_MINAC11":
                    # _MINAC11 requires PADUR1_ and PAFREQ1_ columns
                    required_cols = ["PADUR1_", "PAFREQ1_"]
                    if all(col in column_names for col in required_cols):
                        padur1_idx = column_names.index("PADUR1_")
                        pafreq1_idx = column_names.index("PAFREQ1_")

                        original_data = transformed_data[:, feature_idx].copy()
                        transformed_data[:, feature_idx] = transform_func(
                            transformed_data[:, feature_idx],  # _MINAC11 data
                            transformed_data[:, padur1_idx],  # PADUR1_ data
                            transformed_data[:, pafreq1_idx],  # PAFREQ1_ data
                        )
                    else:
                        missing_cols = [
                            col for col in required_cols if col not in column_names
                        ]
                        if verbose:
                            print(
                                f"    ✗ Missing required columns for {feature_name}: {missing_cols}"
                            )
                        continue
                elif feature_name == "_MINAC21":
                    # _MINAC21 requires PADUR2_ and PAFREQ2_ columns
                    required_cols = ["PADUR2_", "PAFREQ2_"]
                    if all(col in column_names for col in required_cols):
                        padur2_idx = column_names.index("PADUR2_")
                        pafreq2_idx = column_names.index("PAFREQ2_")

                        original_data = transformed_data[:, feature_idx].copy()
                        transformed_data[:, feature_idx] = transform_func(
                            transformed_data[:, feature_idx],  # _MINAC21 data
                            transformed_data[:, padur2_idx],  # PADUR2_ data
                            transformed_data[:, pafreq2_idx],  # PAFREQ2_ data
                        )
                    else:
                        missing_cols = [
                            col for col in required_cols if col not in column_names
                        ]
                        if verbose:
                            print(
                                f"    ✗ Missing required columns for {feature_name}: {missing_cols}"
                            )
                        continue
                elif feature_name == "HHADULT":
                    # HHADULT requires NUMADULT column
                    required_cols = ["NUMADULT"]
                    if all(col in column_names for col in required_cols):
                        numadult_idx = column_names.index("NUMADULT")

                        original_data = transformed_data[:, feature_idx].copy()
                        transformed_data[:, feature_idx] = transform_func(
                            transformed_data[:, feature_idx],  # HHADULT data
                            transformed_data[:, numadult_idx],  # NUMADULT data
                        )
                    else:
                        missing_cols = [
                            col for col in required_cols if col not in column_names
                        ]
                        if verbose:
                            print(
                                f"    ✗ Missing required columns for {feature_name}: {missing_cols}"
                            )
                        continue
                else:
                    # Standard single-column transformation
                    original_data = transformed_data[:, feature_idx].copy()
                    transformed_data[:, feature_idx] = transform_func(
                        transformed_data[:, feature_idx]
                    )

                # Count changes for reporting
                changes = np.sum(original_data != transformed_data[:, feature_idx])
                transformations_applied.append(
                    f"{feature_name}: {changes} values transformed"
                )

                if verbose:
                    print(f"    → {changes} values transformed")

            except Exception as e:
                if verbose:
                    print(f"    ✗ Error transforming {feature_name}: {e}")

    if verbose:
        if transformations_applied:
            print(f"\n✅ Particular transformations completed:")
            for transformation in transformations_applied:
                print(f"  - {transformation}")
            print(f"\n📋 Next: Apply Mae system (FEATURE_CONFIGS) for standard filling")
        else:
            print("ℹ️  No particular case features found in dataset")

    return transformed_data


def get_particular_case_features():
    """
    Get list of all features that have particular case transformations.

    Returns:
        list: Feature names with custom transformations
    """
    return list(TRANSFORMATION_FUNCTIONS.keys())


def add_transformation_function(feature_name, transform_func):
    """
    Add a new transformation function to the registry.

    Args:
        feature_name (str): Name of the feature
        transform_func (callable): Transformation function
    """
    TRANSFORMATION_FUNCTIONS[feature_name] = transform_func
    print(f"✓ Added transformation for {feature_name}")


# Random seed for reproducibility
RANDOM_SEED = 42


def perform_minac21_imputation(
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
    Perform imputation for _MINAC21 feature using particular case transformation
    and standard imputation steps.

    This combines:
    1. Particular case transformation using PADUR2_ and PAFREQ2_ columns
    2. Standard imputation for remaining missing values
    """
    print("=" * 60)
    print("IMPUTING _MINAC21 WITH PARTICULAR CASE HANDLING")
    print("=" * 60)

    # Load data and column names if not provided
    if x_train is None or x_test is None or column_names is None:
        try:
            column_names, train_data = read_csv_with_header(
                f"{output_dir}/x_train_filled.csv"
            )
            _, test_data = read_csv_with_header(f"{output_dir}/x_test_filled.csv")
            train_ids = train_data[:, 0]  # First column is Id
            test_ids = test_data[:, 0]  # First column is Id
            x_train = train_data[:, 1:]  # All columns except Id
            x_test = test_data[:, 1:]  # All columns except Id
            print(f"✓ Data loaded: Train {x_train.shape}, Test {x_test.shape}")
        except Exception as e:
            print(f"❌ ERROR: Could not load data: {e}")
            return None

    # Map feature names to their numeric indices
    # _MINAC21 = feature_296
    # PADUR2_ = feature_292
    # PAFREQ2_ = feature_294
    required_cols = ["feature_296", "feature_292", "feature_294"]
    feature_mapping = {
        "_MINAC21": "feature_296",
        "PADUR2_": "feature_292",
        "PAFREQ2_": "feature_294",
    }
    missing_cols = [col for col in required_cols if col not in column_names]
    if missing_cols:
        print(f"❌ ERROR: Missing required columns: {missing_cols}")
        return None

    # Get column indices using mapped names
    minac21_idx = column_names.index(feature_mapping["_MINAC21"])
    padur2_idx = column_names.index(feature_mapping["PADUR2_"])
    pafreq2_idx = column_names.index(feature_mapping["PAFREQ2_"])

    # Show initial state
    print(f"\n--- Initial State ---")
    train_minac21 = x_train[:, minac21_idx].copy()
    test_minac21 = x_test[:, minac21_idx].copy()

    train_nan = np.sum(np.isnan(train_minac21))
    test_nan = np.sum(np.isnan(test_minac21))

    print(f"_MINAC21 - Initial NaN counts:")
    print(f"Training set: {train_nan} NaN values")
    print(f"Test set: {test_nan} NaN values")

    # Step 1: Apply particular case transformation
    print(f"\n--- Step 1: Applying Particular Case Transformation ---")
    x_train[:, minac21_idx] = transform_minac21(
        x_train[:, minac21_idx], x_train[:, padur2_idx], x_train[:, pafreq2_idx]
    )
    x_test[:, minac21_idx] = transform_minac21(
        x_test[:, minac21_idx], x_test[:, padur2_idx], x_test[:, pafreq2_idx]
    )

    # Check state after transformation
    train_nan_after_transform = np.sum(np.isnan(x_train[:, minac21_idx]))
    test_nan_after_transform = np.sum(np.isnan(x_test[:, minac21_idx]))

    print(f"\nAfter transformation:")
    print(f"Training set: {train_nan_after_transform} NaN values remain")
    print(f"Test set: {test_nan_after_transform} NaN values remain")

    # Step 2: Standard imputation for remaining missing values
    print(f"\n--- Step 2: Standard Imputation ---")

    # Configuration for remaining _MINAC21 values
    minac21_config = [
        {
            "step": "redistribute",
            "values_to_redistribute": [99999],
            "valid_range": (0, 10000),  # Reasonable range for minutes of activity
        },
        {
            "step": "conditional_redistribute",
            "condition_column": "feature_294",  # PAFREQ2_
            "condition_value": lambda x: x > 0,
            "values_to_redistribute": [np.nan],
            "valid_range": (0, 10000),
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "feature_294",  # PAFREQ2_
            "condition_value": 0,
            "target_values": [np.nan],
            "fill_value": 0,
        },
    ]

    # Use the standard imputation function
    from src.cont_features_data_filling import perform_single_feature_imputation

    results = perform_single_feature_imputation(
        feature_mapping["_MINAC21"],
        minac21_config,
        x_train=x_train,
        x_test=x_test,
        column_names=column_names,
        train_ids=train_ids,
        test_ids=test_ids,
        random_state=random_state,
        save_results=save_results,
        output_dir=output_dir,
    )

    return results


def perform_minac11_imputation(
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
    Perform imputation for _MINAC11 feature using particular case transformation
    and standard imputation steps.

    This combines:
    1. Particular case transformation using PADUR1_ and PAFREQ1_ columns
    2. Standard imputation for remaining missing values

    Parameters
    ----------
    x_train : np.ndarray, optional
        Training data. If not provided, will be loaded from files
    x_test : np.ndarray, optional
        Test data. If not provided, will be loaded from files
    column_names : list of str, optional
        Column names. If not provided, will be loaded from files
    train_ids, test_ids : np.ndarray, optional
        IDs for train/test sets. If not provided, will be loaded from files
    random_state : int, optional
        Random seed for reproducibility
    save_results : bool, default=True
        Whether to save the imputed datasets back to files
    output_dir : str, default='dataset_features_removed'
        Directory for saving results

    Returns
    -------
    dict : step_results
        Dictionary with results from each imputation step
    """
    print("=" * 60)
    print("IMPUTING _MINAC11 WITH PARTICULAR CASE HANDLING")
    print("=" * 60)

    # Load data and column names if not provided
    if x_train is None or x_test is None or column_names is None:
        try:
            column_names, train_data = read_csv_with_header(
                f"{output_dir}/x_train_filled.csv"
            )
            _, test_data = read_csv_with_header(f"{output_dir}/x_test_filled.csv")
            train_ids = train_data[:, 0]  # First column is Id
            test_ids = test_data[:, 0]  # First column is Id
            x_train = train_data[:, 1:]  # All columns except Id
            x_test = test_data[:, 1:]  # All columns except Id
            print(f"✓ Data loaded: Train {x_train.shape}, Test {x_test.shape}")
        except Exception as e:
            print(f"❌ ERROR: Could not load data: {e}")
            return None

    # Map feature names to their numeric indices
    # _MINAC11 = feature_305
    # PADUR1_ = feature_303
    # PAFREQ1_ = feature_304
    required_cols = ["feature_295", "feature_291", "feature_293"]
    feature_mapping = {
        "_MINAC11": "feature_295",
        "PADUR1_": "feature_291",  # Original index: 291
        "PAFREQ1_": "feature_293",  # Original index: 293
    }
    missing_cols = [col for col in required_cols if col not in column_names]
    if missing_cols:
        print(f"❌ ERROR: Missing required columns: {missing_cols}")
        return None

    # Get column indices using mapped names
    minac11_idx = column_names.index(feature_mapping["_MINAC11"])
    padur1_idx = column_names.index(feature_mapping["PADUR1_"])
    pafreq1_idx = column_names.index(feature_mapping["PAFREQ1_"])

    # Show initial state
    print(f"\n--- Initial State ---")
    train_minac11 = x_train[:, minac11_idx].copy()
    test_minac11 = x_test[:, minac11_idx].copy()

    train_nan = np.sum(np.isnan(train_minac11))
    test_nan = np.sum(np.isnan(test_minac11))

    print(f"_MINAC11 - Initial NaN counts:")
    print(f"Training set: {train_nan} NaN values")
    print(f"Test set: {test_nan} NaN values")

    # Step 1: Apply particular case transformation
    print(f"\n--- Step 1: Applying Particular Case Transformation ---")
    x_train[:, minac11_idx] = transform_minac11(
        x_train[:, minac11_idx], x_train[:, padur1_idx], x_train[:, pafreq1_idx]
    )
    x_test[:, minac11_idx] = transform_minac11(
        x_test[:, minac11_idx], x_test[:, padur1_idx], x_test[:, pafreq1_idx]
    )

    # Check state after transformation
    train_nan_after_transform = np.sum(np.isnan(x_train[:, minac11_idx]))
    test_nan_after_transform = np.sum(np.isnan(x_test[:, minac11_idx]))

    print(f"\nAfter transformation:")
    print(f"Training set: {train_nan_after_transform} NaN values remain")
    print(f"Test set: {test_nan_after_transform} NaN values remain")

    # Step 2: Standard imputation for remaining missing values
    print(f"\n--- Step 2: Standard Imputation ---")

    # Configuration for remaining _MINAC11 values
    minac11_config = [
        {
            "step": "redistribute",
            "values_to_redistribute": [99999],
            "valid_range": (0, 10000),  # Reasonable range for minutes of activity
        },
        {
            "step": "conditional_redistribute",
            "condition_column": "feature_293",  # PAFREQ1_
            "condition_value": lambda x: x > 0,
            "values_to_redistribute": [np.nan],
            "valid_range": (0, 10000),
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "feature_293",  # PAFREQ1_
            "condition_value": 0,
            "target_values": [np.nan],
            "fill_value": 0,
        },
    ]

    # Use the standard imputation function
    from src.cont_features_data_filling import perform_single_feature_imputation

    results = perform_single_feature_imputation(
        feature_mapping["_MINAC11"],
        minac11_config,
        x_train=x_train,
        x_test=x_test,
        column_names=column_names,
        train_ids=train_ids,
        test_ids=test_ids,
        random_state=random_state,
        save_results=save_results,
        output_dir=output_dir,
    )

    return results


if __name__ == "__main__":
    # Example usage
    results = perform_minac11_imputation(random_state=42)
