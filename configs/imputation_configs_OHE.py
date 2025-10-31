"""
Imputation Configuration File
============================

This file defines four types of data imputation strategies:

1. Value Replacement
   - Direct value substitutions
   - Example: Converting specific codes to NaN

2. Conditional Imputation
   - Values filled based on other features
   - Example: Age-dependent health conditions

3. Weighted Random Imputation
   - Probabilistic filling using observed distributions
   - Example: Categorical responses with known frequencies

4. Complex Feature Engineering
   - Multi-step transformations
   - Example: BMI calculation from height/weight

Each strategy is defined in its respective dictionary with specific formats
and rules for imputation.
"""

import numpy as np

# Value Replacement Configurations
# Format: {column_name: {value_to_replace: new_value}}

value_replacement_configs = {
    "MSCODE": {
        "replacements": [
            (np.nan, 4),
        ]
    },
    "_PAINDX1": {
        "replacements": [
            (9, 3),
        ]
    },
    "_PASTRNG": {
        "replacements": [
            (9, 3),
        ]
    },
    "_FLSHOT6": {
        "replacements": [
            (np.nan, 3),
        ]
    },
    "_PNEUMO2": {
        "replacements": [
            (np.nan, 3),
        ]
    },
}

# Features that should be filled based on logical conditions from other features
# Format: List of dictionaries, each defining a conditional rule

conditional_imputation_configs = [
    {
        "target_column": "_MICHD",
        "condition_columns": ["CVDINFR4", "CVDCRHD4"],
        "target_missing_values": [np.nan],
        "condition_rules": {"CVDINFR4": 2, "CVDCRHD4": 2},
        "logical_operator": "AND",
        "fill_value": 2,
    },
    {
        "target_column": "_ASTHMS1",
        "condition_columns": ["ASTHMA3", "ASTHNOW"],
        "target_missing_values": [9],
        "condition_rules": {"ASTHMA3": 1, "ASTHNOW": 1},
        "logical_operator": "AND",
        "fill_value": 1,
    },
    {
        "target_column": "_ASTHMS1",
        "condition_columns": ["ASTHMA3", "ASTHNOW"],
        "target_missing_values": [9],
        "condition_rules": {"ASTHMA3": 1, "ASTHNOW": 2},
        "logical_operator": "AND",
        "fill_value": 2,
    },
    {
        "target_column": "_SMOKER3",
        "condition_columns": ["SMOKE100", "SMOKEDAY"],
        "target_missing_values": [9],
        "condition_rules": {"SMOKE100": 1, "SMOKEDAY": 1},
        "logical_operator": "AND",
        "fill_value": 1,
    },
    {
        "target_column": "_SMOKER3",
        "condition_columns": ["SMOKE100", "SMOKEDAY"],
        "target_missing_values": [9],
        "condition_rules": {"SMOKE100": 1, "SMOKEDAY": 2},
        "logical_operator": "AND",
        "fill_value": 2,
    },
    {
        "target_column": "_SMOKER3",
        "condition_columns": ["SMOKE100", "SMOKEDAY"],
        "target_missing_values": [9],
        "condition_rules": {"SMOKE100": 1, "SMOKEDAY": 3},
        "logical_operator": "AND",
        "fill_value": 3,
    },
    # Heavy drinking classification based on CDC guidelines
    {
        "target_column": "_RFDRHV5",
        "condition_columns": ["ALCDAY5"],
        "target_missing_values": [9],
        "condition_rules": {"ALCDAY5": 0},
        "logical_operator": "AND",
        "fill_value": 1,
    },
    {
        "target_column": "_RFDRHV5",
        "condition_columns": ["SEX", "_DRNKWEK"],
        "target_missing_values": [9],
        "condition_rules": {"SEX": 1, "_DRNKWEK": lambda x: x <= 14},
        "logical_operator": "AND",
        "fill_value": 1,
    },
    {
        "target_column": "_RFDRHV5",
        "condition_columns": ["SEX", "_DRNKWEK"],
        "target_missing_values": [9],
        "condition_rules": {"SEX": 2, "_DRNKWEK": lambda x: x <= 7},
        "logical_operator": "AND",
        "fill_value": 1,
    },
    {
        "target_column": "_RFDRHV5",
        "condition_columns": ["SEX", "_DRNKWEK"],
        "target_missing_values": [9],
        "condition_rules": {"SEX": 1, "_DRNKWEK": lambda x: x > 14},
        "logical_operator": "AND",
        "fill_value": 2,
    },
    {
        "target_column": "_RFDRHV5",
        "condition_columns": ["SEX", "_DRNKWEK"],
        "target_missing_values": [9],
        "condition_rules": {"SEX": 2, "_DRNKWEK": lambda x: x > 7},
        "logical_operator": "AND",
        "fill_value": 2,
    },
]


# =============================================================================
# 3. WEIGHTED RANDOM IMPUTATION CONFIGURATIONS
# =============================================================================
# Features that should be filled with weighted random choices based on data distribution
# Format: Dictionary mapping column names to their filling configuration

weighted_random_configs = {
    "HIVTST6": {
        "valid_values": [1, 2],
        "probabilities": [0.2858, 0.6776],
        "missing_values": [7, 9, np.nan],
    },
    "PNEUVAC3": {
        "valid_values": [1, 2, 7],
        "probabilities": [0.3947, 0.5146, 0.3],
        "missing_values": [9, np.nan],
    },
    "FLUSHOT6": {
        "valid_values": [1, 2],
        "probabilities": [0.4785, 0.5138],
        "missing_values": [7, 9, np.nan],
    },
    "EXERANY2": {
        "valid_values": [1, 2],
        "probabilities": [0.7921, 0.2646],
        "missing_values": [7, 9, np.nan],
    },
    "_PACAT1": {
        "valid_values": [1, 2, 3, 4],
        "probabilities": [0.2976, 0.1608, 0.1612, 0.2538],
        "missing_values": [9],
    },
    "SMOKE100": {
        "valid_values": [1, 2],
        "probabilities": [0.4314, 0.5609],
        "missing_values": [9, np.nan],
    },
    "USENOW3": {
        "valid_values": [1, 2, 3],
        "probabilities": [0.0176, 0.0142, 0.9636],
        "missing_values": [7, 9, np.nan],
    },
    "DIFFALON": {
        "valid_values": [1, 2],
        "probabilities": [0.0787, 0.9168],
        "missing_values": [7, 9, np.nan],
    },
    "DIFFDRES": {
        "valid_values": [1, 2],
        "probabilities": [0.0438, 0.9534],
        "missing_values": [7, 9, np.nan],
    },
    "DIFFWALK": {
        "valid_values": [1, 2],
        "probabilities": [0.174, 0.8207],
        "missing_values": [7, 9, np.nan],
    },
    "DECIDE": {
        "valid_values": [1, 2],
        "probabilities": [0.0978, 0.8954],
        "missing_values": [7, 9, np.nan],
    },
    "BLIND": {
        "valid_values": [1, 2],
        "probabilities": [0.0495, 0.9466],
        "missing_values": [7, 9, np.nan],
    },
    "USEEQUIP": {
        "valid_values": [1, 2],
        "probabilities": [0.1174, 0.88],
        "missing_values": [7, 9, np.nan],
    },
    "QLACTLM2": {
        "valid_values": [1, 2],
        "probabilities": [0.2473, 0.7453],
        "missing_values": [7, 9, np.nan],
    },
    "INTERNET": {
        "valid_values": [1, 2],
        "probabilities": [0.7864, 0.2104],
        "missing_values": [7, 9, np.nan],
    },
    "EMPLOY1": {
        "valid_values": [1, 2, 3, 4, 5, 6, 7, 8],
        "probabilities": [
            0.4058,
            0.0829,
            0.0217,
            0.0204,
            0.0614,
            0.0262,
            0.3005,
            0.0724,
        ],
        "missing_values": [9],
    },
    "VETERAN3": {
        "valid_values": [1, 2],
        "probabilities": [0.1310, 0.8671],
        "missing_values": [7, 9, np.nan],
    },
    "CPDEMO1": {
        "valid_values": [1, 2],
        "probabilities": [0.8063, 0.1886],
        "missing_values": [7, 9, np.nan],
    },
    "RENTHOM1": {
        "valid_values": [1, 2, 3],
        "probabilities": [0.7173, 0.2289, 0.0467],
        "missing_values": [7, 9],
    },
    "EDUCA": {
        "valid_values": [1, 2, 3, 4, 5, 6],
        "probabilities": [0.0014, 0.0253, 0.0509, 0.2791, 0.2730, 0.3661],
        "missing_values": [9],
    },
    "MARITAL": {
        "valid_values": [1, 2, 3, 4, 5, 6],
        "probabilities": [0.5283, 0.1346, 0.1279, 0.0203, 0.1533, 0.0286],
        "missing_values": [9],
    },
    "DIABETE3": {
        "valid_values": [1, 2, 3, 4],
        "probabilities": [0.1297, 0.0082, 0.8429, 0.0174],
        "missing_values": [7, 9, np.nan],
    },
    "CHCKIDNY": {
        "valid_values": [1, 2],
        "probabilities": [0.0355, 0.961],
        "missing_values": [7, 9],
    },
    "ADDEPEV2": {
        "valid_values": [1, 2],
        "probabilities": [0.1898, 0.8052],
        "missing_values": [7, 9],
    },
    "HAVARTH3": {
        "valid_values": [1, 2],
        "probabilities": [0.3353, 0.6583],
        "missing_values": [7, 9, np.nan],
    },
    "CHCCOPD1": {
        "valid_values": [1, 2],
        "probabilities": [0.0804, 0.9142],
        "missing_values": [7, 9],
    },
    "CHCOCNCR": {
        "valid_values": [1, 2],
        "probabilities": [0.0984, 0.8991],
        "missing_values": [7, 9],
    },
    "CHCSCNCR": {
        "valid_values": [1, 2],
        "probabilities": [0.0946, 0.9025],
        "missing_values": [7, 9, np.nan],
    },
    "CVDSTRK3": {
        "valid_values": [1, 2],
        "probabilities": [0.0414, 0.9557],
        "missing_values": [7, 9],
    },
    "CVDCRHD4": {
        "valid_values": [1, 2],
        "probabilities": [0.0573, 0.9341],
        "missing_values": [7, 9, np.nan],
    },
    "CVDINFR4": {
        "valid_values": [1, 2],
        "probabilities": [0.0577, 0.9373],
        "missing_values": [7, 9],
    },
    "BLOODCHO": {
        "valid_values": [1, 2],
        "probabilities": [0.866, 0.1125],
        "missing_values": [7, 9],
    },
    "BPHIGH4": {
        "valid_values": [1, 2, 3, 4],
        "probabilities": [0.4036, 0.0074, 0.5761, 0.0098],
        "missing_values": [7, 9, np.nan],
    },
    "CHECKUP1": {
        "valid_values": [1, 2, 3, 4, 7, 8],
        "probabilities": [0.7372, 0.1145, 0.0647, 0.0611, 0.0124, 0.0087],
        "missing_values": [9, np.nan],
    },
    "MEDCOST": {
        "valid_values": [1, 2],
        "probabilities": [0.0986, 0.8987],
        "missing_values": [7, 9, np.nan],
    },
    "PERSDOC2": {
        "valid_values": [1, 2, 3],
        "probabilities": [0.7733, 0.0781, 0.1444],
        "missing_values": [7, 9],
    },
    "HLTHPLN1": {
        "valid_values": [1, 2],
        "probabilities": [0.9232, 0.0726],
        "missing_values": [7, 9],
    },
    "GENHLTH": {
        "valid_values": [1, 2, 3, 4, 5],
        "probabilities": [0.1722, 0.3286, 0.3103, 0.1336, 0.0525],
        "missing_values": [7, 9, np.nan],
    },
    "INCOME2": {
        "valid_values": [1, 2, 3, 4, 5, 6, 7, 8],
        "probabilities": [
            0.0421,
            0.0447,
            0.0612,
            0.0739,
            0.0895,
            0.1188,
            0.1327,
            0.2632,
        ],
        "missing_values": [77, 99, np.nan],
    },
    "SEATBELT": {
        "valid_values": [1, 2, 3, 4, 5, 8],
        "probabilities": [0.8693, 0.0687, 0.0274, 0.0113, 0.0154, 0.0022],
        "missing_values": [7, 9, np.nan],
    },
    "CAREGIV1": {
        "valid_values": [1, 2],
        "probabilities": [0.2205, 0.7745],
        "missing_values": [7, 8, 9, np.nan],
    },
    "CRGVEXPT": {
        "valid_values": [1, 2],
        "probabilities": [0.1535, 0.7725],
        "missing_values": [7, 9, np.nan],
    },
    "WTCHSALT": {
        "valid_values": [1, 2],
        "probabilities": [0.6167, 0.3778],
        "missing_values": [7, 9, np.nan],
    },
    "DRADVISE": {
        "valid_values": [1, 2],
        "probabilities": [0.3047, 0.6890],
        "missing_values": [7, 9, np.nan],
    },
    "CVDASPRN": {
        "valid_values": [1, 2],
        "probabilities": [0.299, 0.6995],
        "missing_values": [7, 9, np.nan],
    },
    "TETANUS": {
        "valid_values": [1, 2, 3, 4],
        "probabilities": [0.1589, 0.0725, 0.2438, 0.4248],
        "missing_values": [7, 9, np.nan],
    },
    "HPVADVC2": {
        "valid_values": [1, 2, 3],
        "probabilities": [0.1326, 0.7703, 0.0013],
        "missing_values": [7, 9, np.nan],
    },
    "SCNTMEL1": {
        "valid_values": [1, 2, 3, 4, 5, 8],
        "probabilities": [0.0562, 0.0333, 0.1201, 0.1206, 0.6655, 0.001],
        "missing_values": [7, 9, np.nan],
    },
    "SXORIENT": {
        "valid_values": [1, 2, 3, 4],
        "probabilities": [0.9413, 0.0134, 0.0136, 0.0035],
        "missing_values": [7, 9, np.nan],
    },
    "TRNSGNDR": {
        "valid_values": [1, 2, 3, 4],
        "probabilities": [0.0022, 0.0014, 0.0009, 0.9807],
        "missing_values": [7, 9, np.nan],
    },
    "EMTSUPRT": {
        "valid_values": [1, 2, 3, 4, 5],
        "probabilities": [0.5357, 0.2807, 0.1029, 0.0297, 0.0345],
        "missing_values": [7, 9, np.nan],
    },
    "LSATISFY": {
        "valid_values": [1, 2, 3, 4],
        "probabilities": [0.4521, 0.4848, 0.0412, 0.0112],
        "missing_values": [7, 9, np.nan],
    },
    "MISTMNT": {
        "valid_values": [1, 2],
        "probabilities": [0.1469, 0.8488],
        "missing_values": [7, 9, np.nan],
    },
    "ADANXEV": {
        "valid_values": [1, 2],
        "probabilities": [0.1619, 0.8317],
        "missing_values": [7, 9, np.nan],
    },
    "ASTHMA3": {
        "valid_values": [1, 2],
        "probabilities": [0.1346, 0.862],
        "missing_values": [7, 9],
    },
    "_PRACE1": {
        "valid_values": [1, 2, 3, 4, 5, 6, 7, 8],
        "probabilities": [
            0.8227,
            0.0835,
            0.0198,
            0.0247,
            0.0066,
            0.0202,
            0.0026,
            0.00002,
        ],
        "missing_values": [77, 99],
    },
    "_MRACE1": {
        "valid_values": [1, 2, 3, 4, 5, 6, 7],
        "probabilities": [0.8136, 0.0813, 0.018, 0.0229, 0.0039, 0.0196, 0.0206],
        "missing_values": [77, 99],
    },
    "_HISPANC": {
        "valid_values": [1, 2],
        "probabilities": [0.0811, 0.9095],
        "missing_values": [9],
    },
}

# Complex Feature Engineering Configurations
# Format: {column_name: [{step: operation, ...parameters}]}

complex_engineering_configs = {
    "SMOKDAY2": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],
            "valid_values": [1, 2, 3],
            "probabilities": [0.2366, 0.0977, 0.6639],
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "SMOKE100",
            "condition_value": 2,
            "fill_value": 4,
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],
            "valid_values": [1, 2, 3],
            "probabilities": [0.2366, 0.0977, 0.6639],
        },
    ],
    "STOPSMK2": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],
            "valid_values": [1, 2],
            "probabilities": [0.5683, 0.4283],
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "SMOKDAY2",
            "condition_value": 3,
            "fill_value": 3,
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],
            "valid_values": [1, 2],
            "probabilities": [0.5683, 0.4283],
        },
    ],
    "LASTSMK2": [
        {
            "step": "weighted_random",
            "target_values": [77, 99],
            "valid_values": [1, 2, 3, 4, 5, 6, 7, 8],
            "probabilities": [
                0.021,
                0.0192,
                0.0202,
                0.0331,
                0.1284,
                0.0999,
                0.6642,
                0.0074,
            ],
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "SMOKE100",
            "condition_value": 2,
            "fill_value": 9,
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],
            "valid_values": [1, 2, 3, 4, 5, 6, 7, 8],
            "probabilities": [
                0.021,
                0.0192,
                0.0202,
                0.0331,
                0.1284,
                0.0999,
                0.6642,
                0.0074,
            ],
        },
    ],
    "HIVTSTD3": [
        {
            "step": "weighted_random",
            "target_values": [777777, 999999],
            "valid_values": [1, 2, 3, 4],
            "probabilities": [0.1137, 0.2003, 0.6858, 0.0002],
        },
        {
            "step": "conditional_weighted_random",
            "target_values": [np.nan],
            "condition_column": "HIVTST6",
            "condition_value": 1,
            "valid_values": [1, 2, 3, 4],
            "probabilities": [0.1137, 0.2003, 0.6858, 0.0002],
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "HIVTST6",
            "condition_value": 2,
            "fill_value": 5,
        },
    ],
    "FLSHTMY2": [
        {
            "step": "weighted_random",
            "target_values": [777777, 999999],
            "valid_values": [1, 2],
            "probabilities": [0.0873, 0.9127],
        },
        {
            "step": "conditional_weighted_random",
            "target_values": [np.nan],
            "condition_column": "FLUSHOT6",
            "condition_value": 1,
            "valid_values": [1, 2],
            "probabilities": [0.0873, 0.9127],
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "FLUSHOT6",
            "condition_value": 2,
            "fill_value": 3,
        },
    ],
    "PREGNANT": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],
            "valid_values": [1, 2],
            "probabilities": [0.0374, 0.9550],
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "SEX",
            "condition_value": 1,
            "fill_value": 2,
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "_AGE80",
            "condition_value": lambda x: x >= 45,
            "fill_value": 2,
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],
            "valid_values": [1, 2],
            "probabilities": [0.0374, 0.9550],
        },
    ],
    "TOLDHI2": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],
            "valid_values": [1, 2],
            "probabilities": [0.4184, 0.5722],
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "BLOODCHO",
            "condition_value": 2,
            "fill_value": 3,
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],
            "valid_values": [1, 2],
            "probabilities": [0.4184, 0.5722],
        },
    ],
    "CHOLCHK": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],
            "valid_values": [1, 2, 3, 4],
            "probabilities": [0.7673, 0.114, 0.0679, 0.0354],
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "BLOODCHO",
            "condition_value": 2,
            "fill_value": 5,
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],
            "valid_values": [1, 2, 3, 4],
            "probabilities": [0.7673, 0.114, 0.0679, 0.0354],
        },
    ],
    "BPMEDS": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],
            "valid_values": [1, 2],
            "probabilities": [0.8364, 0.1618],
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "BPHIGH4",
            "condition_value": lambda x: x >= 2,
            "fill_value": 3,
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],
            "valid_values": [1, 2],
            "probabilities": [0.8364, 0.1618],
        },
    ],
    "EXRACT11": [
        {
            "step": "weighted_random",
            "target_values": [77, 99],
            "valid_values": list(range(1, 77)) + [98],
            "probabilities": [
                0.0005,
                0.0237,
                0.0001,
                0.0001,
                0.0060,
                0.0227,
                0.0248,
                0.0006,
                0.0021,
                0.0011,
                0.0158,
                0.0003,
                0.0005,
                0.0068,
                0.0164,
                0.0008,
                0.0003,
                0.0550,
                0.0124,
                0.0056,
                0.0001,
                0.0093,
                0.0009,
                0.0016,
                0.0008,
                0.0005,
                0.0001,
                0.0094,
                0.0001,
                0.0002,
                0.0041,
                0.0001,
                0.0002,
                0.0020,
                0.0008,
                0.0006,
                0.0664,
                0.0006,
                0.0001,
                0.0009,
                0.0003,
                0.0001,
                0.0005,
                0.0005,
                0.0001,
                0.0001,
                0.0003,
                0.0029,
                0.0019,
                0.0004,
                0.0030,
                0.0017,
                0.0001,
                0.0032,
                0.0000,
                0.0004,
                0.0094,
                0.0053,
                0.0001,
                0.0010,
                0.0039,
                0.0004,
                0.0015,
                0.5455,
                0.0003,
                0.0003,
                0.0369,
                0.0003,
                0.0099,
                0.0018,
                0.0018,
                0.0025,
                0.0068,
                0.0016,
                0.0002,
                0.0164,
                0.0407,
            ],
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "EXERANY2",
            "condition_value": 2,
            "fill_value": 78,
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],
            "valid_values": list(range(1, 77)) + [98],
            "probabilities": [
                0.0005,
                0.0237,
                0.0001,
                0.0001,
                0.0060,
                0.0227,
                0.0248,
                0.0006,
                0.0021,
                0.0011,
                0.0158,
                0.0003,
                0.0005,
                0.0068,
                0.0164,
                0.0008,
                0.0003,
                0.0550,
                0.0124,
                0.0056,
                0.0001,
                0.0093,
                0.0009,
                0.0016,
                0.0008,
                0.0005,
                0.0001,
                0.0094,
                0.0001,
                0.0002,
                0.0041,
                0.0001,
                0.0002,
                0.0020,
                0.0008,
                0.0006,
                0.0664,
                0.0006,
                0.0001,
                0.0009,
                0.0003,
                0.0001,
                0.0005,
                0.0005,
                0.0001,
                0.0001,
                0.0003,
                0.0029,
                0.0019,
                0.0004,
                0.0030,
                0.0017,
                0.0001,
                0.0032,
                0.0000,
                0.0004,
                0.0094,
                0.0053,
                0.0001,
                0.0010,
                0.0039,
                0.0004,
                0.0015,
                0.5455,
                0.0003,
                0.0003,
                0.0369,
                0.0003,
                0.0099,
                0.0018,
                0.0018,
                0.0025,
                0.0068,
                0.0016,
                0.0002,
                0.0164,
                0.0407,
            ],
        },
    ],
    "EXRACT21": [
        {
            "step": "weighted_random",
            "target_values": [77, 99],
            "valid_values": list(range(1, 77)) + [88] + [98],
            "probabilities": [
                0.0007,
                0.0158,
                0.0001,
                0.0002,
                0.0068,
                0.0201,
                0.0275,
                0.0013,
                0.0030,
                0.0009,
                0.0187,
                0.0005,
                0.0013,
                0.0088,
                0.0119,
                0.0025,
                0.0004,
                0.0466,
                0.0093,
                0.0056,
                0.0001,
                0.0090,
                0.0006,
                0.0021,
                0.0009,
                0.0007,
                0.0001,
                0.0063,
                0.0001,
                0.0003,
                0.0080,
                0.0001,
                0.0001,
                0.0030,
                0.0008,
                0.0010,
                0.0303,
                0.0005,
                0.0004,
                0.0013,
                0.0001,
                0.0001,
                0.0004,
                0.0006,
                0.0001,
                0.0001,
                0.0005,
                0.0029,
                0.0019,
                0.0004,
                0.0030,
                0.0017,
                0.0001,
                0.0032,
                0.0000,
                0.0004,
                0.0094,
                0.0053,
                0.0001,
                0.0012,
                0.0039,
                0.0004,
                0.0015,
                0.1393,
                0.0002,
                0.0621,
                0.001,
                0.001,
                0.0099,
                0.0008,
                0.0008,
                0.006,
                0.0049,
                0.001,
                0.0003,
                0.0345,
                0.3148,
                0.0515,
            ],
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "EXERANY2",
            "condition_value": 2,
            "fill_value": 78,
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],
            "valid_values": list(range(1, 77)) + [88] + [98],
            "probabilities": [
                0.0007,
                0.0158,
                0.0001,
                0.0002,
                0.0068,
                0.0201,
                0.0275,
                0.0013,
                0.0030,
                0.0009,
                0.0187,
                0.0005,
                0.0013,
                0.0088,
                0.0119,
                0.0025,
                0.0004,
                0.0466,
                0.0093,
                0.0056,
                0.0001,
                0.0090,
                0.0006,
                0.0021,
                0.0009,
                0.0007,
                0.0001,
                0.0063,
                0.0001,
                0.0003,
                0.0080,
                0.0001,
                0.0001,
                0.0030,
                0.0008,
                0.0010,
                0.0303,
                0.0005,
                0.0004,
                0.0013,
                0.0001,
                0.0001,
                0.0004,
                0.0006,
                0.0001,
                0.0001,
                0.0005,
                0.0029,
                0.0019,
                0.0004,
                0.0030,
                0.0017,
                0.0001,
                0.0032,
                0.0000,
                0.0004,
                0.0094,
                0.0053,
                0.0001,
                0.0012,
                0.0039,
                0.0004,
                0.0015,
                0.1393,
                0.0002,
                0.0621,
                0.001,
                0.001,
                0.0099,
                0.0008,
                0.0008,
                0.006,
                0.0049,
                0.001,
                0.0003,
                0.0345,
                0.3148,
                0.0515,
            ],
        },
    ],
    "LMTJOIN3": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],
            "valid_values": [1, 2],
            "probabilities": [0.4911, 0.4968],
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "HAVARTH3",
            "condition_value": 2,
            "fill_value": 2,
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],
            "valid_values": [1, 2],
            "probabilities": [0.4911, 0.4968],
        },
    ],
    "ARTHDIS2": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],
            "valid_values": [1, 2],
            "probabilities": [0.3103, 0.6598],
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "HAVARTH3",
            "condition_value": 2,
            "fill_value": 2,
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],
            "valid_values": [1, 2],
            "probabilities": [0.3103, 0.6598],
        },
    ],
    "ARTHSOCL": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],
            "valid_values": [1, 2, 3],
            "probabilities": [0.1830, 0.2379, 0.5699],
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "HAVARTH3",
            "condition_value": 2,
            "fill_value": 3,
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],
            "valid_values": [1, 2, 3],
            "probabilities": [0.1830, 0.2379, 0.5699],
        },
    ],
    "IMFVPLAC": [
        {
            "step": "weighted_random",
            "target_values": [77, 99],
            "valid_values": list(range(1, 12)),
            "probabilities": [
                0.3801,
                0.0231,
                0.0922,
                0.0115,
                0.2539,
                0.0578,
                0.0007,
                0.1395,
                0.0302,
                0.0004,
                0.0072,
            ],
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "FLUSHOT6",
            "condition_value": 2,
            "fill_value": 12,
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],
            "valid_values": list(range(1, 12)),
            "probabilities": [
                0.3801,
                0.0231,
                0.0922,
                0.0115,
                0.2539,
                0.0578,
                0.0007,
                0.1395,
                0.0302,
                0.0004,
                0.0072,
            ],
        },
    ],
    "WHRTST10": [
        {
            "step": "weighted_random",
            "target_values": [77, 99],
            "valid_values": list(range(1, 10)),
            "probabilities": [
                0.4586,
                0.0354,
                0.0944,
                0.2141,
                0.0107,
                0.0046,
                0.0248,
                0.1201,
                0.0174,
            ],
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "HIVTST6",
            "condition_value": 2,
            "fill_value": 10,
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],
            "valid_values": list(range(1, 10)),
            "probabilities": [
                0.4586,
                0.0354,
                0.0944,
                0.2141,
                0.0107,
                0.0046,
                0.0248,
                0.1201,
                0.0174,
            ],
        },
    ],
    "PDIABTST": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],
            "valid_values": [1, 2],
            "probabilities": [0.6127, 0.3406],
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "DIABETE3",
            "condition_value": 1,
            "fill_value": 3,
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],
            "valid_values": [1, 2],
            "probabilities": [0.6127, 0.3406],
        },
    ],
    "PREDIAB1": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],
            "valid_values": [1, 2, 3],
            "probabilities": [0.1033, 0.0119, 0.8807],
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "DIABETE3",
            "condition_value": 1,
            "fill_value": 4,
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],
            "valid_values": [1, 2, 3],
            "probabilities": [0.1033, 0.0119, 0.8807],
        },
    ],
    "INSULIN": [
        {
            "step": "weighted_random",
            "target_values": [9],
            "valid_values": [1, 2],
            "probabilities": [0.3263, 0.6725],
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "DIABETE3",
            "condition_value": lambda x: x >= 2,
            "fill_value": 2,
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],
            "valid_values": [1, 2],
            "probabilities": [0.3263, 0.6725],
        },
    ],
    "EYEEXAM": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],
            "valid_values": [1, 2, 3, 4, 8],
            "probabilities": [0.2124, 0.4979, 0.1268, 0.1169, 0.0308],
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "DIABETE3",
            "condition_value": lambda x: x >= 2,
            "fill_value": 5,
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],
            "valid_values": [1, 2, 3, 4, 8],
            "probabilities": [0.2124, 0.4979, 0.1268, 0.1169, 0.0308],
        },
    ],
    "DIABEYE": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],
            "valid_values": [1, 2],
            "probabilities": [0.1812, 0.8055],
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "DIABETE3",
            "condition_value": lambda x: x >= 2,
            "fill_value": 3,
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],
            "valid_values": [1, 2],
            "probabilities": [0.1812, 0.8055],
        },
    ],
    "DIABEDU": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],
            "valid_values": [1, 2],
            "probabilities": [0.5562, 0.4394],
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "DIABETE3",
            "condition_value": lambda x: x >= 2,
            "fill_value": 3,
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],
            "valid_values": [1, 2],
            "probabilities": [0.5562, 0.4394],
        },
    ],
    "CRGVREL1": [
        {
            "step": "weighted_random",
            "target_values": [77, 99],
            "valid_values": list(range(1, 16)),
            "probabilities": [
                0.2224,
                0.0814,
                0.0403,
                0.0152,
                0.0865,
                0.103,
                0.0711,
                0.0015,
                0.0347,
                0.0481,
                0.0357,
                0.0121,
                0.0089,
                0.0722,
                0.1602,
            ],
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "CAREGIV1",
            "condition_value": 2,
            "fill_value": 16,
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],
            "valid_values": list(range(1, 16)),
            "probabilities": [
                0.2224,
                0.0814,
                0.0403,
                0.0152,
                0.0865,
                0.103,
                0.0711,
                0.0015,
                0.0347,
                0.0481,
                0.0357,
                0.0121,
                0.0089,
                0.0722,
                0.1602,
            ],
        },
    ],
    "CRGVLNG1": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],
            "valid_values": [1, 2, 3, 4, 5],
            "probabilities": [0.1825, 0.1241, 0.1814, 0.2084, 0.2889],
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "CAREGIV1",
            "condition_value": 2,
            "fill_value": 6,
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],
            "valid_values": [1, 2, 3, 4, 5],
            "probabilities": [0.1825, 0.1241, 0.1814, 0.2084, 0.2889],
        },
    ],
    "CRGVHRS1": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],
            "valid_values": [1, 2, 3, 4],
            "probabilities": [0.5468, 0.1221, 0.0985, 0.1771],
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "CAREGIV1",
            "condition_value": 2,
            "fill_value": 5,
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],
            "valid_values": [1, 2, 3, 4],
            "probabilities": [0.5468, 0.1221, 0.0985, 0.1771],
        },
    ],
    "CRGVPRB1": [
        {
            "step": "weighted_random",
            "target_values": [77, 99],
            "valid_values": list(range(1, 14)),
            "probabilities": [
                0.0615,
                0.0051,
                0.0798,
                0.0401,
                0.0974,
                0.0333,
                0.0555,
                0.0702,
                0.0015,
                0.0382,
                0.0239,
                0.0031,
                0.4438,
            ],
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "CAREGIV1",
            "condition_value": 2,
            "fill_value": 14,
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],
            "valid_values": list(range(1, 14)),
            "probabilities": [
                0.0615,
                0.0051,
                0.0798,
                0.0401,
                0.0974,
                0.0333,
                0.0555,
                0.0702,
                0.0015,
                0.0382,
                0.0239,
                0.0031,
                0.4438,
            ],
        },
    ],
    "CRGVPERS": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],
            "valid_values": [1, 2],
            "probabilities": [0.5062, 0.4877],
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "CAREGIV1",
            "condition_value": 2,
            "fill_value": 3,
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],
            "valid_values": [1, 2],
            "probabilities": [0.5062, 0.4877],
        },
    ],
    "CRGVHOUS": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],
            "valid_values": [1, 2],
            "probabilities": [0.7755, 0.2185],
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "CAREGIV1",
            "condition_value": 2,
            "fill_value": 3,
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],
            "valid_values": [1, 2],
            "probabilities": [0.7755, 0.2185],
        },
    ],
    "CRGVMST2": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],
            "valid_values": [1, 2, 3, 4, 5, 6],
            "probabilities": [0.0134, 0.0833, 0.0268, 0.0194, 0.027, 0.7996],
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "CAREGIV1",
            "condition_value": 2,
            "fill_value": 8,
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],
            "valid_values": [1, 2, 3, 4, 5, 6],
            "probabilities": [0.0134, 0.0833, 0.0268, 0.0194, 0.027, 0.7996],
        },
    ],
    "CIMEMLOS": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],
            "valid_values": [1, 2],
            "probabilities": [0.1063, 0.8848],
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "_AGE80",
            "condition_value": lambda x: x <= 44,
            "fill_value": 3,
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],
            "valid_values": [1, 2],
            "probabilities": [0.1063, 0.8848],
        },
    ],
    "CDHOUSE": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],
            "valid_values": [1, 2, 3, 4, 5],
            "probabilities": [0.0701, 0.063, 0.2408, 0.1362, 0.4673],
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "_AGE80",
            "condition_value": lambda x: x <= 44,
            "fill_value": 6,
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "CIMEMLOS",
            "condition_value": 2,
            "fill_value": 8,
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],
            "valid_values": [1, 2, 3, 4, 5],
            "probabilities": [0.0701, 0.063, 0.2408, 0.1362, 0.4673],
        },
    ],
    "CDASSIST": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],
            "valid_values": [1, 2, 3, 4, 5],
            "probabilities": [0.062, 0.0508, 0.2142, 0.1495, 0.5108],
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "_AGE80",
            "condition_value": lambda x: x <= 44,
            "fill_value": 6,
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],
            "condition_column": "CIMEMLOS",
            "condition_value": 2,
            "fill_value": 8,
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],
            "valid_values": [1, 2, 3, 4, 5],
            "probabilities": [0.062, 0.0508, 0.2142, 0.1495, 0.5108],
        },
    ],  # FROM HERE, THE DOCUMENTATION ARE WRONG BECAUSE ENTRIES WERE COPY PASTED AND MODIFIED WITHOUT CHANGING THE DOC.
    "CDHELP": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2, 3, 4, 5],
            "probabilities": [
                0.4108,
                0.2032,
                0.2429,
                0.0735,
                0.0613,
            ],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "_AGE80",
            "condition_value": lambda x: x <= 44,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 6,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "CIMEMLOS",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 8,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "CDASSIST",
            "condition_value": 5,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 10,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2, 3, 4, 5],
            "probabilities": [0.4108, 0.2032, 0.2429, 0.0735, 0.0613],
        },
    ],
    "CDSOCIAL": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2, 3, 4, 5],
            "probabilities": [
                0.0948,
                0.0562,
                0.1851,
                0.1456,
                0.4987,
            ],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "_AGE80",
            "condition_value": lambda x: x <= 44,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 6,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "CIMEMLOS",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 8,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2, 3, 4, 5],
            "probabilities": [0.0948, 0.0562, 0.1851, 0.1456, 0.4987],
        },
    ],
    "CDDISCUS": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2],
            "probabilities": [0.4441, 0.5469],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "_AGE80",
            "condition_value": lambda x: x <= 44,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 3,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "CIMEMLOS",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 4,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2],
            "probabilities": [0.4441, 0.5469],
        },
    ],
    "HAREHAB1": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2],
            "probabilities": [0.3986, 0.5856],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "CVDINFR4",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 3,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2],
            "probabilities": [0.3986, 0.5856],
        },
    ],
    "STREHAB1": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2],
            "probabilities": [0.3746, 0.6138],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "CVDSTRK3",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 3,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2],
            "probabilities": [0.3746, 0.6138],
        },
    ],
    "ASPUNSAF": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2, 3],
            "probabilities": [0.0734, 0.0433, 0.8741],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "CVDASPRN",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 3,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2, 3],
            "probabilities": [0.0734, 0.0433, 0.8741],
        },
    ],
    "RLIVPAIN": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2],
            "probabilities": [0.207, 0.7882],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "CVDASPRN",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 3,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2],
            "probabilities": [0.207, 0.7882],
        },
    ],
    "RDUCHART": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2],
            "probabilities": [0.8299, 0.1436],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "CVDASPRN",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 3,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2],
            "probabilities": [0.8299, 0.1436],
        },
    ],
    "RDUCSTRK": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2],
            "probabilities": [0.6604, 0.2647],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "CVDASPRN",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 3,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2],
            "probabilities": [0.6604, 0.2647],
        },
    ],
    "ARTTODAY": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2, 3, 4],
            "probabilities": [
                0.2277,
                0.4258,
                0.247,
                0.0929,
            ],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "HAVARTH3",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 5,  # Fill with 5 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2, 3, 4],
            "probabilities": [0.2277, 0.4258, 0.247, 0.0929],
        },
    ],
    "ARTHWGT": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2],
            "probabilities": [0.3616, 0.6312],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "HAVARTH3",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 3,  # Fill with 5 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2],
            "probabilities": [0.3616, 0.6312],
        },
    ],
    "ARTHEXER": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2],
            "probabilities": [0.5718, 0.4158],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "HAVARTH3",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 3,  # Fill with 5 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2],
            "probabilities": [0.5718, 0.4158],
        },
    ],
    "ARTHEDU": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2],
            "probabilities": [0.1305, 0.8645],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "HAVARTH3",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 3,  # Fill with 5 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2],
            "probabilities": [0.1305, 0.8645],
        },
    ],
    "SHINGLE2": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2],
            "probabilities": [0.2805, 0.7028],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "_AGE80",
            "condition_value": lambda x: x < 50,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 2,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2],
            "probabilities": [0.2805, 0.7028],
        },
    ],
    "HADMAM": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2],
            "probabilities": [0.8129, 0.1837],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "SEX",
            "condition_value": 1,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 3,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2],
            "probabilities": [0.8129, 0.1837],
        },
    ],
    "HOWLONG": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2, 3, 4, 5],
            "probabilities": [
                0.6155,
                0.1723,
                0.0677,
                0.0515,
                0.0836,
            ],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "SEX",
            "condition_value": 1,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 6,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "HADMAM",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 8,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2, 3, 4, 5],
            "probabilities": [0.6155, 0.1723, 0.0677, 0.0515, 0.0836],
        },
    ],
    "HADPAP2": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2],
            "probabilities": [0.9329, 0.06],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "SEX",
            "condition_value": 1,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 3,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2],
            "probabilities": [0.9329, 0.06],
        },
    ],
    "LASTPAP2": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2, 3, 4, 5],
            "probabilities": [
                0.4040,
                0.1851,
                0.10,
                0.0737,
                0.2136,
            ],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "SEX",
            "condition_value": 1,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 6,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "HADPAP2",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 8,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2, 3, 4, 5],
            "probabilities": [0.4040, 0.1851, 0.10, 0.0737, 0.2136],
        },
    ],
    "HPVTEST": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2],
            "probabilities": [0.3032, 0.4085],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "SEX",
            "condition_value": 1,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 3,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2],
            "probabilities": [0.3032, 0.4085],
        },
    ],
    "HPLSTTST": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2, 3, 4, 5],
            "probabilities": [
                0.4452,
                0.1742,
                0.0886,
                0.0696,
                0.1678,
            ],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "SEX",
            "condition_value": 1,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 6,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "HPVTEST",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 8,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2, 3, 4, 5],
            "probabilities": [0.4452, 0.1742, 0.0886, 0.0696, 0.1678],
        },
    ],
    "HADHYST2": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2],
            "probabilities": [0.2804, 0.7149],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "SEX",
            "condition_value": 1,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 3,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "PREGNANT",
            "condition_value": 1,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 2,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2],
            "probabilities": [0.2804, 0.7149],
        },
    ],
    "PROFEXAM": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2],
            "probabilities": [0.8742, 0.1222],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "SEX",
            "condition_value": 1,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 3,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2],
            "probabilities": [0.8742, 0.1222],
        },
    ],
    "LENGEXAM": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2, 3, 4, 5],
            "probabilities": [
                0.614,
                0.1703,
                0.0685,
                0.0458,
                0.0906,
            ],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "SEX",
            "condition_value": 1,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 6,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "PROFEXAM",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 8,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2, 3, 4, 5],
            "probabilities": [0.614, 0.1703, 0.0685, 0.0458, 0.0906],
        },
    ],
    "BLDSTOOL": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2],
            "probabilities": [0.3638, 0.6271],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "_AGE80",
            "condition_value": lambda x: x < 50,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 3,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2],
            "probabilities": [0.3638, 0.6271],
        },
    ],
    "LSTBLDS3": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2, 3, 4, 5],
            "probabilities": [
                0.2692,
                0.1357,
                0.0977,
                0.1187,
                0.3488,
            ],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "_AGE80",
            "condition_value": lambda x: x < 50,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 6,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "BLDSTOOL",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 8,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2, 3, 4, 5],
            "probabilities": [0.2692, 0.1357, 0.0977, 0.1187, 0.3488],
        },
    ],
    "HADSIGM3": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2],
            "probabilities": [0.7247, 0.2698],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "_AGE80",
            "condition_value": lambda x: x < 50,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 3,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2],
            "probabilities": [0.7247, 0.2698],
        },
    ],
    "HADSGCO1": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2],
            "probabilities": [0.0388, 0.9404],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "_AGE80",
            "condition_value": lambda x: x < 50,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 3,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "HADSIGM3",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 4,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2],
            "probabilities": [0.0388, 0.9404],
        },
    ],
    "LASTSIG3": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2, 3, 4, 5, 6],
            "probabilities": [
                0.2151,
                0.1565,
                0.1469,
                0.2128,
                0.1879,
                0.067,
            ],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "_AGE80",
            "condition_value": lambda x: x < 50,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 7,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "HADSIGM3",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 10,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2, 3, 4, 5, 6],
            "probabilities": [0.2151, 0.1565, 0.1469, 0.2128, 0.1879, 0.067],
        },
    ],
    "PCPSAAD2": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2],
            "probabilities": [0.5861, 0.3711],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "_AGE80",
            "condition_value": lambda x: x < 40,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 3,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "SEX",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 3,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2],
            "probabilities": [0.5861, 0.3711],
        },
    ],
    "PCPSADI1": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2],
            "probabilities": [0.2749, 0.6778],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "_AGE80",
            "condition_value": lambda x: x < 40,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 3,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "SEX",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 3,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2],
            "probabilities": [0.2749, 0.6778],
        },
    ],
    "PCPSARE1": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2],
            "probabilities": [0.5223, 0.4366],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "_AGE80",
            "condition_value": lambda x: x < 40,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 3,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "SEX",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 3,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2],
            "probabilities": [0.5223, 0.4366],
        },
    ],
    "PSATEST1": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2],
            "probabilities": [0.5495, 0.4057],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "_AGE80",
            "condition_value": lambda x: x < 40,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 3,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "SEX",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 3,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2],
            "probabilities": [0.5495, 0.4057],
        },
    ],
    "PSATIME": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2, 3, 4, 5],
            "probabilities": [
                0.6293,
                0.1439,
                0.0658,
                0.0572,
                0.0812,
            ],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "_AGE80",
            "condition_value": lambda x: x < 40,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 6,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "SEX",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 6,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "PSATEST1",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 8,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2, 3, 4, 5],
            "probabilities": [0.6293, 0.1439, 0.0658, 0.0572, 0.0812],
        },
    ],
    "PCPSARS1": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2, 3, 4, 5],
            "probabilities": [
                0.7406,
                0.0705,
                0.0485,
                0.046,
                0.087,
            ],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "_AGE80",
            "condition_value": lambda x: x < 40,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 6,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "SEX",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 6,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "PSATEST1",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 8,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2, 3, 4, 5],
            "probabilities": [0.7406, 0.0705, 0.0485, 0.046, 0.087],
        },
    ],
    "PCPSADE1": [
        {
            "step": "weighted_random",
            "target_values": [4, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2, 3],
            "probabilities": [0.2466, 0.3358, 0.3016],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "_AGE80",
            "condition_value": lambda x: x < 40,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 4,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "SEX",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 4,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "PSATEST1",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 5,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2, 3],
            "probabilities": [0.2466, 0.3358, 0.3016],
        },
    ],
    "SCNTMNY1": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2, 3, 4, 5, 8],
            "probabilities": [
                0.0879,
                0.0409,
                0.1511,
                0.1614,
                0.5401,
                0.0151,
            ],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "RENTHOM1",
            "condition_value": 3,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 8,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2, 3, 4, 5, 8],
            "probabilities": [0.0879, 0.0409, 0.1511, 0.1614, 0.5401, 0.0151],
        },
    ],
    "SCNTPAID": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2, 3, 4],
            "probabilities": [
                0.3741,
                0.4704,
                0.0864,
                0.0568,
            ],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "EMPLOY1",
            "condition_value": lambda x: x >= 3,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 5,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2, 3, 4],
            "probabilities": [0.3741, 0.4704, 0.0864, 0.0568],
        },
    ],
    "RCSGENDR": [
        {
            "step": "weighted_random",
            "target_values": [9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2],
            "probabilities": [0.4916, 0.4611],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "CHILDREN",
            "condition_value": 0,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 3,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2],
            "probabilities": [0.4916, 0.4611],
        },
    ],
    "RCSRLTN2": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2, 3, 4, 5, 6],
            "probabilities": [
                0.7828,
                0.0842,
                0.0117,
                0.0456,
                0.0232,
                0.0148,
            ],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "CHILDREN",
            "condition_value": 0,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 8,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2, 3, 4, 5, 6],
            "probabilities": [0.7828, 0.0842, 0.0117, 0.0456, 0.0232, 0.0148],
        },
    ],
    "CASTHDX2": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2],
            "probabilities": [0.1244, 0.8279],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "CHILDREN",
            "condition_value": 0,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 3,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2],
            "probabilities": [0.1244, 0.8279],
        },
    ],
    "CASTHNO2": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2],
            "probabilities": [0.6549, 0.3209],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "CHILDREN",
            "condition_value": 0,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 3,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "CASTHDX2",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 4,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2],
            "probabilities": [0.6549, 0.3209],
        },
    ],
    "_CPRACE": [
        {
            "step": "weighted_random",
            "target_values": [77, 99],  # Step 1: Replace 7 and 9 first
            "valid_values": [1, 2, 3, 4, 5, 6, 7],
            "probabilities": [
                0.7185,
                0.0899,
                0.0266,
                0.0283,
                0.0033,
                0.0271,
                0.0469,
            ],  # Adjust based on your data
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "CHILDREN",
            "condition_value": 0,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 8,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2, 3, 4, 5, 6, 7],
            "probabilities": [0.7185, 0.0899, 0.0266, 0.0283, 0.0033, 0.0271, 0.0469],
        },
    ],
    "_RFHLTH": [
        {
            "step": "conditional_fill",
            "target_values": [9],  # Step 2: Replace NaN conditionally
            "condition_column": "GENHLTH",
            "condition_value": lambda x: x <= 3,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 1,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [9],  # Step 2: Replace NaN conditionally
            "condition_column": "GENHLTH",
            "condition_value": lambda x: x >= 4,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 2,  # Fill with 2 (no - logical)
        },
    ],
    "_RFHYPE5": [
        {
            "step": "conditional_fill",
            "target_values": [9],  # Step 2: Replace NaN conditionally
            "condition_column": "BPHIGH4",
            "condition_value": lambda x: x >= 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 1,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [9],  # Step 2: Replace NaN conditionally
            "condition_column": "BPHIGH4",
            "condition_value": 1,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 2,  # Fill with 2 (no - logical)
        },
    ],
    "_CHOLCHK": [
        {
            "step": "conditional_fill",
            "target_values": [9],  # Step 2: Replace NaN conditionally
            "condition_column": "BLOODCHO",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 3,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [9],  # Step 2: Replace NaN conditionally
            "condition_column": "CHOLCHK",
            "condition_value": 4,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 2,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [9],  # Step 2: Replace NaN conditionally
            "condition_column": "CHOLCHK",
            "condition_value": lambda x: x <= 3,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 1,  # Fill with 2 (no - logical)
        },
    ],
    "_RFCHOL": [
        {
            "step": "conditional_fill",
            "target_values": [9, np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "TOLDHI2",
            "condition_value": 1,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 2,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [9, np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "TOLDHI2",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 1,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "BLOODCHO",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 3,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [9, np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2],
            "probabilities": [0.5722, 0.4184],
        },
    ],
    "_MICHD": [
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "CVDINFR4",
            "condition_value": 1,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 1,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "CVDCRHD4",
            "condition_value": 1,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 1,  # Fill with 2 (no - logical)
        },
    ],
    "ASTHNOW": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2],
            "probabilities": [0.6733, 0.2963],
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "ASTHMA3",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 3,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2],
            "probabilities": [0.6733, 0.2963],  # Fill with 2 (no - logical)
        },
    ],
    "_ASTHMS1": [
        {
            "step": "conditional_fill",
            "target_values": [9],  # Step 2: Replace NaN conditionally
            "condition_column": "ASTHMA3",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 3,  # Fill with 2 (no - logical)
        }
    ],
    "_BMI5CAT": [
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "_BMI5",
            "condition_value": lambda x: x
            < 18.5,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 1,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "_BMI5",
            "condition_value": lambda x: 18.5
            <= x
            < 25,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 2,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "_BMI5",
            "condition_value": lambda x: 25 <= x < 30,
            "fill_value": 3,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "_BMI5",
            "condition_value": lambda x: x >= 30,
            "fill_value": 4,  # Fill with 2 (no - logical)
        },
    ],
    "_SMOKER3": [
        {
            "step": "conditional_fill",
            "target_values": [9],  # Step 2: Replace NaN conditionally
            "condition_column": "SMOKE100",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 4,  # Fill with 2 (no - logical)
        }
    ],
    "DRNKANY5": [
        {
            "step": "conditional_fill",
            "target_values": [7, 9],  # Step 2: Replace NaN conditionally
            "condition_column": "DROCDY3_",
            "condition_value": 0,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 2,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [7, 9],  # Step 2: Replace NaN conditionally
            "condition_column": "DROCDY3_",
            "condition_value": lambda x: x >= 1,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 1,  # Fill with 2 (no - logical)
        },
    ],
    "ACTIN11_": [
        {
            "step": "conditional_weighted_fill",
            "target_values": [np.nan],  # Values to replace
            "condition_column": "EXERANY2",  # Column to check condition on
            "condition_value": 1,  # Condition (can be value or function)
            "valid_values": [0, 1, 2],  # Values to choose from
            "probabilities": [0.0503, 0.6182, 0.3315],  # Probabilities (28.5%, 71.5%)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "EXERANY2",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 0,  # Fill with 2 (no - logical)
        },
    ],
    "ACTIN21_": [
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "EXRACT21",
            "condition_value": 88,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 0,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "EXERANY2",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 0,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [0, 1, 2],
            "probabilities": [0.4036, 0.3194, 0.2769],  # Fill with 2 (no - logical)
        },
    ],
    "_FLSHOT6": [
        {
            "step": "weighted_random",
            "target_values": [9],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2],
            "probabilities": [0.5488, 0.3313],  # Fill with 2 (no - logical)
        }
    ],
    "_PNEUMO2": [
        {
            "step": "conditional_fill",
            "target_values": [9],  # Step 2: Replace NaN conditionally
            "condition_column": "PNEUVAC3",
            "condition_value": 1,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 1,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [9],  # Step 2: Replace NaN conditionally
            "condition_column": "PNEUVAC3",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 2,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [9],  # Step 2: Replace NaN conditionally
            "condition_column": "PNEUVAC3",
            "condition_value": 7,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 4,  # Fill with 2 (no - logical)
        },
    ],
    "ASATTACK": [
        {
            "step": "conditional_fill",
            "target_values": [7, np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "ASTHMA3",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 2,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [7, np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "ASTHNOW",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 2,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [7, np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2],
            "probabilities": [0.483, 0.50],  # Fill with 2 (no - logical)
        },
    ],
    "ASYMPTOM": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2, 3, 4, 5, 8],
            "probabilities": [
                0.1778,
                0.2137,
                0.1139,
                0.1139,
                0.0515,
                0.2855,
            ],  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "ASTHMA3",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 6,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "ASTHNOW",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 6,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2, 3, 4, 5, 8],
            "probabilities": [
                0.1778,
                0.2137,
                0.1139,
                0.1139,
                0.0515,
                0.2855,
            ],  # Fill with 2 (no - logical)
        },
    ],
    "ASNOSLEP": [
        {
            "step": "weighted_random",
            "target_values": [7],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2, 3, 4, 5, 8],
            "probabilities": [
                0.1532,
                0.1028,
                0.0503,
                0.0635,
                0.1072,
                0.488,
            ],  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "ASYMPTOM",
            "condition_value": 8,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 8,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "ASTHMA3",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 6,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "ASTHNOW",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 6,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2, 3, 4, 5, 8],
            "probabilities": [
                0.1532,
                0.1028,
                0.0503,
                0.0635,
                0.1072,
                0.488,
            ],  # Fill with 2 (no - logical)
        },
    ],
    "ASTHMED3": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2, 3, 8],
            "probabilities": [
                0.2003,
                0.0532,
                0.3803,
                0.3349,
            ],  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "ASTHMA3",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 4,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "ASTHNOW",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 4,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2, 3, 8],
            "probabilities": [
                0.2003,
                0.0532,
                0.3803,
                0.3349,
            ],  # Fill with 2 (no - logical)
        },
    ],
    "ASINHALR": [
        {
            "step": "weighted_random",
            "target_values": [7, 9],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2, 3, 4, 5, 6, 8],
            "probabilities": [
                0.2801,
                0.0767,
                0.0454,
                0.072,
                0.0188,
                0.0078,
                0.4773,
            ],  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "ASTHMA3",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 10,  # Fill with 2 (no - logical)
        },
        {
            "step": "conditional_fill",
            "target_values": [np.nan],  # Step 2: Replace NaN conditionally
            "condition_column": "ASTHNOW",
            "condition_value": 2,  # When DIABETE3 = 2 (younger than 65)
            "fill_value": 10,  # Fill with 2 (no - logical)
        },
        {
            "step": "weighted_random",
            "target_values": [np.nan],  # Step 3: Remaining NaN with weighted random
            "valid_values": [1, 2, 3, 4, 5, 6, 8],
            "probabilities": [
                0.2801,
                0.0767,
                0.0454,
                0.072,
                0.0188,
                0.0078,
                0.4773,
            ],  # Fill with 2 (no - logical)
        },
    ],
    "_RFBING5": [
        {
            "step": "conditional_fill",
            "target_values": [9],  # Replace 9 (Don't know) for non-binge drinkers
            "condition_column": "DRNK3GE5",
            "condition_value": 0,  # No binge drinking days
            "fill_value": 1,  # No binge drinking (1 = No)
        },
        {
            "step": "conditional_fill",
            "target_values": [9],  # Replace 9 (Don't know) for binge drinkers
            "condition_column": "DRNK3GE5",
            "condition_value": lambda x: x > 0,  # Has binge drinking days
            "fill_value": 2,  # Yes, binge drinking (2 = Yes)
        },
    ],
}
