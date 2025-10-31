# Open the CSV file and read just the header line
data_path = "data/processed/dataset_features_removed/x_train_filled.csv"
with open(data_path, "r") as file:
    header = file.readline().strip()
    feature_names = header.split(",")

print("Column names in the dataset:")
print("-" * 50)
for i, name in enumerate(feature_names):
    print(f"Column {i}: {name}")

# Load data (skip header)
import numpy as np

x_train_raw_ = np.genfromtxt(data_path, delimiter=",", skip_header=1)

# List of indices to check
indices = [
    1,
    2,
    3,
    4,
    8,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    36,
    37,
    38,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    59,
    60,
    61,
    62,
    63,
    64,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    102,
    103,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
    124,
    126,
    127,
    128,
    129,
    130,
    131,
    132,
    133,
    134,
    135,
    136,
    137,
    138,
    139,
    140,
    141,
    142,
    143,
    144,
    145,
    146,
    147,
    148,
    149,
    150,
    152,
    153,
    154,
    155,
    156,
    157,
    158,
    159,
    167,
    169,
    170,
    171,
    172,
    173,
    174,
    175,
    176,
    177,
    178,
    179,
    180,
    182,
    187,
    188,
    190,
    192,
    201,
    202,
    207,
    208,
    214,
    215,
    216,
    217,
    218,
]

print("\nUnique values for selected features:")
print("-" * 50)
for idx in indices:
    if idx < len(feature_names):
        col_name = feature_names[idx]
        unique_values = np.unique(x_train_raw_[:, idx])
        print(f"{col_name} (Index {idx}): {unique_values}")
    else:
        print(f"Index {idx} out of bounds for columns.")

# Print total number of unique values across all selected OHE features
total_unique = 0
for idx in indices:
    if idx < len(feature_names):
        unique_values = np.unique(x_train_raw_[:, idx])
        total_unique += len(unique_values)
print(
    f"\nTotal number of unique values across all selected OHE features: {total_unique}"
)

# List of normalization indices
normalization_indices = [
    5,
    6,
    7,
    9,
    10,
    11,
    35,
    39,
    55,
    56,
    57,
    58,
    65,
    77,
    78,
    79,
    80,
    81,
    101,
    104,
    105,
    106,
    107,
    125,
    151,
    160,
    161,
    162,
    163,
    164,
    165,
    166,
    168,
    181,
    183,
    184,
    185,
    186,
    189,
    191,
    193,
    194,
    195,
    196,
    197,
    198,
    199,
    200,
    203,
    204,
    205,
    206,
    209,
    210,
    211,
    212,
    213,
]

print("\nUnique values for normalization features:")
print("-" * 50)
for idx in normalization_indices:
    if idx < len(feature_names):
        col_name = feature_names[idx]
        unique_values = np.unique(x_train_raw_[:, idx])
        print(f"{col_name} (Index {idx}): {unique_values}")
    else:
        print(f"Index {idx} out of bounds for columns.")

# Print total number of unique values across all normalization features
total_unique_norm = 0
for idx in normalization_indices:
    if idx < len(feature_names):
        unique_values = np.unique(x_train_raw_[:, idx])
        total_unique_norm += len(unique_values)
print(
    f"\nTotal number of unique values across all normalization features: {total_unique_norm}"
)
