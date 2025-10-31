import numpy as np

FEATURE_CONFIGS = {
    "PHYSHLTH": {
        "simple_replacements": {88: 0},
        "redistribute_config": {
            "values_to_redistribute": [77, 99, np.nan],
            "valid_range": (0, 30),
        },
    },
    "MENTHLTH": {
        "simple_replacements": {88: 0},
        "redistribute_config": {
            "values_to_redistribute": [77, 99, np.nan],
            "valid_range": (0, 30),
        },
    },
    "POORHLTH": {
        "simple_replacements": {88: 0},
        "redistribute_config": {
            "values_to_redistribute": [77, 99, np.nan],
            "valid_range": (0, 30),
        },
    },
    "NUMPHON2": {
        "simple_replacements": {"nan": 0},
        "redistribute_config": {
            "values_to_redistribute": [7, 9],
            "valid_range": (0, 6),
        },
    },
    "STRFREQ_": {
        "redistribute_config": {
            "values_to_redistribute": [99, np.nan],
            "valid_range": (0, 98.9),
        }
    },
    "MAXVO2_": {
        "redistribute_config": {
            "values_to_redistribute": [999],
            "valid_range": (0, 50.1),
        }
    },
    "FC60_": {
        "redistribute_config": {
            "values_to_redistribute": [999],
            "valid_range": (0, 8.590),
        }
    },
    "HTM4": {
        "redistribute_config": {
            "values_to_redistribute": [np.nan],
            "valid_range": (0.91, 2.44),
        }
    },
    "HHADULT": {
        "redistribute_config": {
            "values_to_redistribute": [77, 99, np.nan],
            "valid_range": (1, 60),
        }
    },
    "_BMI5": {
        "redistribute_config": {
            "values_to_redistribute": [np.nan],
            "valid_range": (12, 100),
        }
    },
    "WTKG3": {
        "redistribute_config": {
            "values_to_redistribute": [np.nan, 999.99],
            "valid_range": (22.68, 289.85),
        }
    },
    "ADEAT1": {
        "simple_replacements": {88: 0},
        "redistribute_config": {
            "values_to_redistribute": [77, 99, np.nan],
            "valid_range": (0, 14),
        },
    },
    "ADFAIL": {
        "simple_replacements": {88: 0},
        "redistribute_config": {
            "values_to_redistribute": [77, 99, np.nan],
            "valid_range": (1, 14),
        },
    },
    "ADTHINK": {
        "simple_replacements": {88: 0},
        "redistribute_config": {
            "values_to_redistribute": [77, 99, np.nan],
            "valid_range": (1, 14),
        },
    },
    "ADMOVE": {
        "simple_replacements": {88: 0},
        "redistribute_config": {
            "values_to_redistribute": [77, 99, np.nan],
            "valid_range": (1, 14),
        },
    },
    "ADENERGY": {
        "simple_replacements": {88: 0},
        "redistribute_config": {
            "values_to_redistribute": [77, 99, np.nan],
            "valid_range": (1, 14),
        },
    },
    "ADSLEEP": {
        "simple_replacements": {88: 0},
        "redistribute_config": {
            "values_to_redistribute": [77, 99, np.nan],
            "valid_range": (1, 14),
        },
    },
    "ADDOWN": {
        "simple_replacements": {88: 0},
        "redistribute_config": {
            "values_to_redistribute": [77, 88, 99, np.nan],
            "valid_range": (1, 14),
        },
    },
    "ADPLEASR": {
        "simple_replacements": {88: 0},
        "redistribute_config": {
            "values_to_redistribute": [77, 99, np.nan],
            "valid_range": (1, 14),
        },
    },
    "CHILDREN": {
        "simple_replacements": {88: 0},
        "redistribute_config": {
            "values_to_redistribute": [99, np.nan],
            "valid_range": (0, 52),
        },
    },
    "NUMWOMEN": {
        "redistribute_config": {
            "values_to_redistribute": [np.nan],
            "valid_range": (0, 10),
        }
    },
    "NUMMEN": {
        "redistribute_config": {
            "values_to_redistribute": [np.nan],
            "valid_range": (0, 18),
        }
    },
    "DROCDY3_": {
        "redistribute_config": {
            "values_to_redistribute": [900],
            "valid_range": (0, 100),
        }
    },
    "FTJUDA1_": {
        "redistribute_config": {
            "values_to_redistribute": [np.nan],
            "valid_range": (0, 75),
        }
    },
    "FRUTDA1_": {
        "redistribute_config": {
            "values_to_redistribute": [np.nan],
            "valid_range": (0, 75),
        }
    },
    "BEANDAY_": {
        "redistribute_config": {
            "values_to_redistribute": [np.nan],
            "valid_range": (0, 50),
        }
    },
    "GRENDAY_": {
        "redistribute_config": {
            "values_to_redistribute": [np.nan],
            "valid_range": (0, 75),
        }
    },
    "ORNGDAY_": {
        "redistribute_config": {
            "values_to_redistribute": [np.nan],
            "valid_range": (0, 31),
        }
    },
    "VEGEDA1_": {
        "redistribute_config": {
            "values_to_redistribute": [np.nan],
            "valid_range": (0, 75),
        }
    },
    "ALCDAY5": {
        "simple_replacements": {888: 0},
        "redistribute_config": {
            "values_to_redistribute": [777, 999, np.nan],
            "valid_range": (0, 30),
        },
    },
}


# =============================================================================
# CONDITIONAL FEATURE CONFIGS
# =============================================================================
# Conditional redistribution: Fill based on other feature values
# Similar to your conditional_imputation_configs but for continuous/ordinal data

CONDITIONAL_FEATURE_CONFIGS = []


# =============================================================================
# COMPLEX MULTI-STEP ENGINEERING CONFIGS
# =============================================================================
# Multi-step preprocessing: Similar to your complex_engineering_configs
# Each feature can have multiple ordered steps

COMPLEX_ENGINEERING_CONFIGS = {
    "JOINPAIN": [
        {
            "step": "redistribute",
            "values_to_redistribute": [77, 99],  # Replace 77 and 99 with distribution
            "valid_range": (0, 10),
        },
        {
            "step": "conditional_redistribute",
            "condition_column": "HAVARTH3",
            "condition_value": 1,  # Has arthritis
            "values_to_redistribute": [
                np.nan
            ],  # Replace NaN values for arthritis patients
            "valid_range": (0, 10),  # Same distribution as general population
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "HAVARTH3",
            "condition_value": 2,  # No arthritis
            "target_values": [np.nan],  # Replace remaining NaN values
            "fill_value": 0,  # No joint pain for non-arthritis patients
        },
    ],
    # BELOW THE DOCUMENTATION IS WRONG BECAUSE ENTRIES WERE COPY PASTED AND MODIFIED WITHOUT CHANGING THE DOC.
    "DOCTDIAB": [
        {
            "step": "simple_replacement",
            "replacements": {88: 0},  # Don't know → No visits
        },
        {
            "step": "redistribute",
            "values_to_redistribute": [77, 99],  # Don't know (77) and Refused (99)
            "valid_range": (0, 98),  # Based on actual data range
        },
        {
            "step": "conditional_redistribute",
            "condition_column": "DIABETE3",
            "condition_value": 1,  # Has diabetes
            "values_to_redistribute": [
                np.nan
            ],  # Replace NaN values for diabetic patients
            "valid_range": (0, 98),  # Same distribution for diabetic patients
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "DIABETE3",
            "condition_value": lambda x: (x == 2)
            | (x == 3)
            | (x == 4),  # No diabetes, pre-diabetes, gestational only
            "target_values": [np.nan],  # Replace NaN values for non-diabetic patients
            "fill_value": 0,  # No diabetes doctor visits for non-diabetic patients
        },
    ],
    "CHKHEMO3": [
        {
            "step": "simple_replacement",
            "replacements": {88: 0, 98: 0},  # Don't know and Not applicable → No test
        },
        {
            "step": "redistribute",
            "values_to_redistribute": [77, 99],  # Don't know (77) and Refused (99)
            "valid_range": (
                0,
                76,
            ),  # Based on actual data range: 1-76, extended to 0-76
        },
        {
            "step": "conditional_redistribute",
            "condition_column": "DIABETE3",
            "condition_value": 1,  # Has diabetes
            "values_to_redistribute": [
                np.nan
            ],  # Replace NaN values for diabetic patients
            "valid_range": (0, 76),  # Same distribution for diabetic patients
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "DIABETE3",
            "condition_value": lambda x: (x == 2)
            | (x == 3)
            | (x == 4),  # No diabetes, pre-diabetes, gestational only
            "target_values": [np.nan],  # Replace NaN values for non-diabetic patients
            "fill_value": 0,  # No hemoglobin A1c tests for non-diabetic patients
        },
    ],
    "ASERVIST": [
        {
            "step": "simple_replacement",
            "replacements": {88: 0},  # Don't know → No emergency visits
        },
        {
            "step": "redistribute",
            "values_to_redistribute": [98],  # Not applicable → redistribute
            "valid_range": (
                0,
                87,
            ),  # Based on actual data range: 1-87, extended to 0-87
        },
        {
            "step": "conditional_redistribute",
            "condition_column": "ASATTACK",
            "condition_value": 1,  # Has asthma attack
            "values_to_redistribute": [
                np.nan
            ],  # Replace NaN values for asthma patients
            "valid_range": (
                0,
                87,
            ),  # Distribution for asthma patients (includes those with and without attacks)
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "ASTHMA3",
            "condition_value": 2,  # No asthma
            "target_values": [np.nan],  # Replace NaN values for non-asthma patients
            "fill_value": 0,  # No emergency visits for non-asthma patients
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "ASTHNOW",
            "condition_value": 2,  # No current asthma
            "target_values": [np.nan],  # Replace remaining NaN values
            "fill_value": 0,  # No emergency visits for no current asthma
        },
    ],
    "ASDRVIST": [
        {
            "step": "simple_replacement",
            "replacements": {88: 0},  # Don't know → No doctor visits
        },
        {
            "step": "redistribute",
            "values_to_redistribute": [98, 99],  # Not applicable → redistribute
            "valid_range": (
                0,
                87,
            ),  # Based on actual data range: 1-87, extended to 0-87
        },
        {
            "step": "conditional_redistribute",
            "condition_column": "ASATTACK",
            "condition_value": 1,  # Has asthma
            "values_to_redistribute": [
                np.nan
            ],  # Replace NaN values for asthma patients
            "valid_range": (
                0,
                87,
            ),  # Distribution for asthma patients (includes those with and without attacks)
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "ASTHMA3",
            "condition_value": 2,  # No asthma
            "target_values": [np.nan],  # Replace NaN values for non-asthma patients
            "fill_value": 0,  # No doctor visits for non-asthma patients
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "ASTHNOW",
            "condition_value": 2,  # No current asthma
            "target_values": [np.nan],  # Replace remaining NaN values
            "fill_value": 0,  # No doctor visits for no current asthma
        },
    ],
    "ASRCHKUP": [
        {
            "step": "simple_replacement",
            "replacements": {88: 0},  # Don't know → No time since checkup (recent)
        },
        {
            "step": "redistribute",
            "values_to_redistribute": [98, 99],  # Not applicable (98) and Refused (99)
            "valid_range": (
                0,
                87,
            ),  # Based on actual data range: 1-365 days, extended to 0-365
        },
        {
            "step": "conditional_redistribute",
            "condition_column": "ASTHNOW",
            "condition_value": 1,  # Has current asthma
            "values_to_redistribute": [
                np.nan
            ],  # Replace NaN values for current asthma patients
            "valid_range": (0, 87),  # Distribution for current asthma patients
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "ASTHMA3",
            "condition_value": 2,  # No asthma history
            "target_values": [np.nan],  # Replace NaN values for non-asthma patients
            "fill_value": 0,  # No checkups for non-asthma patients
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "ASTHNOW",
            "condition_value": 2,  # No current asthma
            "target_values": [np.nan],  # Replace remaining NaN values
            "fill_value": 0,  # No recent checkups for no current asthma
        },
    ],
    "ASACTLIM": [
        {
            "step": "simple_replacement",
            "replacements": {888: 0},  # Don't know → No activity limitation
        },
        {
            "step": "redistribute",
            "values_to_redistribute": [777, 999],  # Don't know (777) and Refused (999)
            "valid_range": (
                0,
                365,
            ),  # Based on actual data range: 1-365 days, extended to 0-365
        },
        {
            "step": "conditional_redistribute",
            "condition_column": "ASTHNOW",
            "condition_value": 1,  # Has current asthma
            "values_to_redistribute": [
                np.nan
            ],  # Replace NaN values for current asthma patients
            "valid_range": (0, 365),  # Distribution for current asthma patients
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "ASTHMA3",
            "condition_value": 2,  # No asthma history
            "target_values": [np.nan],  # Replace NaN values for non-asthma patients
            "fill_value": 0,  # No activity limitation for non-asthma patients
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "ASTHNOW",
            "condition_value": 2,  # No current asthma
            "target_values": [np.nan],  # Replace remaining NaN values
            "fill_value": 0,  # No activity limitation for no current asthma
        },
    ],
    "HPVADSHT": [
        {
            "step": "redistribute",
            "values_to_redistribute": [77, 99],  # Don't know (77) and Refused (99)
            "valid_range": (0, 3),  # Based on actual data range: 1-3, extended to 0-3
        },
        {
            "step": "conditional_redistribute",
            "condition_column": "HPVADVC2",
            "condition_value": 1,  # HPV vaccine recommended
            "values_to_redistribute": [np.nan],  # Replace NaN values when recommended
            "valid_range": (
                0,
                3,
            ),  # Distribution for those who were recommended vaccine
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "HPVADVC2",
            "condition_value": lambda x: (x == 2)
            | (x == 3),  # Not recommended or Don't know
            "target_values": [np.nan],  # Replace NaN values for those not recommended
            "fill_value": 0,  # No HPV shots for those not recommended
        },
    ],
    "SCNTWRK1": [
        {
            "step": "simple_replacement",
            "replacements": {98: 0},  # Not applicable → 0 hours worked
        },
        {
            "step": "redistribute",
            "values_to_redistribute": [97, 99],  # Don't know (97) and Refused (99)
            "valid_range": (
                0,
                96,
            ),  # Based on actual data range: 1-96 hours, extended to 0-96
        },
        {
            "step": "conditional_redistribute",
            "condition_column": "EMPLOY1",
            "condition_value": lambda x: (x == 1)
            | (x == 2),  # Employed for wages or Self-employed
            "values_to_redistribute": [
                np.nan
            ],  # Replace NaN values for employed people
            "valid_range": (0, 96),  # Distribution for employed people
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "EMPLOY1",
            "condition_value": lambda x: x
            > 2,  # Unemployed, homemaker, student, retired, unable to work
            "target_values": [np.nan],  # Replace NaN values for non-employed people
            "fill_value": 0,  # No work hours for non-employed people
        },
    ],
    "METVL11_": [
        {
            "step": "conditional_redistribute",
            "condition_column": "EXERANY2",
            "condition_value": 1,  # Yes (exercises)
            "values_to_redistribute": [
                np.nan
            ],  # Replace NaN values for people who exercise
            "valid_range": (0, 12.8),  # Based on actual data range: 0-12.8 MET values
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "EXERANY2",
            "condition_value": 2,  # No (does not exercise)
            "target_values": [
                np.nan
            ],  # Replace remaining NaN values for non-exercisers
            "fill_value": 0,  # No metabolic equivalent value for non-exercisers
        },
    ],
    "METVL21_": [
        {
            "step": "conditional_redistribute",
            "condition_column": "EXRACT21",
            "condition_value": lambda x: x < 88,  # Valid exercise activity codes
            "values_to_redistribute": [
                np.nan
            ],  # Replace NaN values for people with valid exercise activities
            "valid_range": (0, 12.8),  # Based on actual data range: 0-12.8 MET values
        },
        {
            "step": "conditional_redistribute",
            "condition_column": "EXRACT21",
            "condition_value": 98,  # Don't know/Not sure - use distribution
            "values_to_redistribute": [
                np.nan
            ],  # Replace NaN values for people with unknown exercise activity
            "valid_range": (0, 12.8),  # Based on actual data range: 0-12.8 MET values
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "EXRACT21",
            "condition_value": 88,  # No exercise activity (88 = None)
            "target_values": [
                np.nan
            ],  # Replace NaN values for people with no exercise activity
            "fill_value": 0,  # No metabolic equivalent value for no exercise
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "EXERANY2",
            "condition_value": 2,  # No (does not exercise)
            "target_values": [
                np.nan
            ],  # Replace remaining NaN values for non-exercisers
            "fill_value": 0,  # No metabolic equivalent value for non-exercisers
        },
    ],
    "PADUR1_": [
        {
            "step": "conditional_redistribute",
            "condition_column": "EXERANY2",
            "condition_value": 1,  # Yes (exercises)
            "values_to_redistribute": [
                np.nan
            ],  # Replace NaN values for people who exercise
            "valid_range": (
                0,
                600,
            ),  # Based on actual data range: 1-599 minutes, extended to 0-600
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "EXERANY2",
            "condition_value": 2,  # No (does not exercise)
            "target_values": [
                np.nan
            ],  # Replace remaining NaN values for non-exercisers
            "fill_value": 0,  # No physical activity duration for non-exercisers
        },
    ],
    "PADUR2_": [
        {
            "step": "conditional_redistribute",
            "condition_column": "EXRACT21",
            "condition_value": lambda x: x < 88,  # Valid exercise activity codes
            "values_to_redistribute": [
                np.nan
            ],  # Replace NaN values for people with valid exercise activities
            "valid_range": (
                0,
                600,
            ),  # Based on actual data range: 1-599 minutes, extended to 0-600
        },
        {
            "step": "conditional_redistribute",
            "condition_column": "EXRACT21",
            "condition_value": 98,  # Don't know/Not sure - use distribution
            "values_to_redistribute": [
                np.nan
            ],  # Replace NaN values for people with unknown exercise activity
            "valid_range": (
                0,
                600,
            ),  # Based on actual data range: 1-599 minutes, extended to 0-600
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "EXRACT21",
            "condition_value": 88,  # No exercise activity (88 = None)
            "target_values": [
                np.nan
            ],  # Replace NaN values for people with no exercise activity
            "fill_value": 0,  # No physical activity duration for no exercise
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "EXERANY2",
            "condition_value": 2,  # No (does not exercise)
            "target_values": [
                np.nan
            ],  # Replace remaining NaN values for non-exercisers
            "fill_value": 0,  # No physical activity duration for non-exercisers
        },
    ],
    "PAFREQ1_": [
        {
            "step": "redistribute",
            "values_to_redistribute": [
                99
            ],  # Replace 99 (don't know/refused) with distribution
            "valid_range": (
                0,
                90,
            ),  # Based on actual data range: 0.233-90.0, extended to 0-90
        },
        {
            "step": "conditional_redistribute",
            "condition_column": "EXERANY2",
            "condition_value": 1,  # Yes (exercises)
            "values_to_redistribute": [
                np.nan
            ],  # Replace NaN values for people who exercise
            "valid_range": (
                0,
                90,
            ),  # Based on actual data range: 0.233-90.0, extended to 0-90
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "EXERANY2",
            "condition_value": 2,  # No (does not exercise)
            "target_values": [
                np.nan
            ],  # Replace remaining NaN values for non-exercisers
            "fill_value": 0,  # No physical activity frequency for non-exercisers
        },
    ],
    "PAFREQ2_": [
        {
            "step": "redistribute",
            "values_to_redistribute": [
                99
            ],  # Replace 99 (don't know/refused) with distribution
            "valid_range": (
                0,
                76,
            ),  # Based on actual data range: 0.233-76.0, extended to 0-76
        },
        {
            "step": "conditional_redistribute",
            "condition_column": "EXRACT21",
            "condition_value": lambda x: x < 88,  # Valid exercise activity codes
            "values_to_redistribute": [
                np.nan
            ],  # Replace NaN values for people with valid exercise activities
            "valid_range": (
                0,
                76,
            ),  # Based on actual data range: 0.233-76.0, extended to 0-76
        },
        {
            "step": "conditional_redistribute",
            "condition_column": "EXRACT21",
            "condition_value": 98,  # Don't know/Not sure - use distribution
            "values_to_redistribute": [
                np.nan
            ],  # Replace NaN values for people with unknown exercise activity
            "valid_range": (
                0,
                76,
            ),  # Based on actual data range: 0.233-76.0, extended to 0-76
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "EXRACT21",
            "condition_value": 88,  # No exercise activity (88 = None)
            "target_values": [
                np.nan
            ],  # Replace NaN values for people with no exercise activity
            "fill_value": 0,  # No physical activity frequency for no exercise
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "EXERANY2",
            "condition_value": 2,  # No (does not exercise)
            "target_values": [
                np.nan
            ],  # Replace remaining NaN values for non-exercisers
            "fill_value": 0,  # No physical activity frequency for non-exercisers
        },
    ],
    "PAVIG11_": [
        {
            "step": "conditional_redistribute",
            "condition_column": "ACTIN11_",
            "condition_value": 2,  # Has vigorous activity
            "values_to_redistribute": [
                np.nan
            ],  # Replace NaN values for people with vigorous activity
            "valid_range": (0, 14400),  # Based on actual data range: 0-14400 minutes
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "ACTIN11_",
            "condition_value": lambda x: x < 2,  # No vigorous activity
            "target_values": [
                np.nan
            ],  # Replace NaN values for people with no vigorous activity
            "fill_value": 0,  # No vigorous activity time for people who don't do vigorous activities
        },
    ],
    "PAVIG21_": [
        {
            "step": "conditional_redistribute",
            "condition_column": "ACTIN21_",
            "condition_value": 2,  # Has vigorous activity
            "values_to_redistribute": [
                np.nan
            ],  # Replace NaN values for people with vigorous activity
            "valid_range": (0, 19200),  # Based on actual data range: 0-19200 minutes
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "ACTIN21_",
            "condition_value": lambda x: x < 2,  # No vigorous activity
            "target_values": [
                np.nan
            ],  # Replace NaN values for people with no vigorous activity
            "fill_value": 0,  # No vigorous activity time for people who don't do vigorous activities
        },
    ],
    "AVEDRNK2": [
        {
            "step": "redistribute",
            "values_to_redistribute": [77, 99],  # Don't know (77) and Refused (99)
            "valid_range": (
                0,
                76,
            ),  # Based on actual data range: 1-76 drinks, extended to 0-76
        },
        {
            "step": "conditional_redistribute",
            "condition_column": "ALCDAY5",
            "condition_value": lambda x: x > 0,  # Has drinking days
            "values_to_redistribute": [
                np.nan
            ],  # Replace NaN values for people who drink
            "valid_range": (0, 76),  # Distribution for people who drink alcohol
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "ALCDAY5",
            "condition_value": 0,  # No drinking days
            "target_values": [np.nan],  # Replace remaining NaN values for non-drinkers
            "fill_value": 0,  # No drinks per occasion for non-drinkers
        },
    ],
    "DRNK3GE5": [
        {
            "step": "simple_replacement",
            "replacements": {88: 0},  # Don't know → 0 (no binge drinking days)
        },
        {
            "step": "redistribute",
            "values_to_redistribute": [77, 99],  # Don't know (77) and Refused (99)
            "valid_range": (
                0,
                76,
            ),  # Based on actual data range: 1-76 days, extended to 0-76
        },
        {
            "step": "conditional_redistribute",
            "condition_column": "ALCDAY5",
            "condition_value": lambda x: x > 0,  # Has drinking days
            "values_to_redistribute": [
                np.nan
            ],  # Replace NaN values for people who drink
            "valid_range": (0, 76),  # Distribution for people who drink alcohol
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "ALCDAY5",
            "condition_value": 0,  # No drinking days
            "target_values": [np.nan],  # Replace remaining NaN values for non-drinkers
            "fill_value": 0,  # No binge drinking days for non-drinkers
        },
    ],
    "MAXDRNKS": [
        {
            "step": "redistribute",
            "values_to_redistribute": [77, 99],  # Don't know (77) and Refused (99)
            "valid_range": (
                0,
                76,
            ),  # Based on actual data range: 1-76 drinks, extended to 0-76
        },
        {
            "step": "conditional_redistribute",
            "condition_column": "ALCDAY5",
            "condition_value": lambda x: x > 0,  # Has drinking days
            "values_to_redistribute": [
                np.nan
            ],  # Replace NaN values for people who drink
            "valid_range": (0, 76),  # Distribution for people who drink alcohol
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "ALCDAY5",
            "condition_value": 0,  # No drinking days
            "target_values": [np.nan],  # Replace remaining NaN values for non-drinkers
            "fill_value": 0,  # No maximum drinks for non-drinkers
        },
    ],
    "LONGWTCH": [
        {
            "step": "redistribute",
            "values_to_redistribute": [
                555,
                777,
                999,
            ],  # Long time/Don't know, Don't know and Refused
            "valid_range": (0, 36500),  # 0 days to ~100 years after transformation
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "WTCHSALT",
            "condition_value": 2,  # No - does not watch salt intake
            "target_values": [
                np.nan
            ],  # Replace remaining NaN values for non-salt watchers
            "fill_value": 0,  # No time watching salt for those who don't watch salt
        },
        {
            "step": "redistribute",
            "values_to_redistribute": [
                np.nan
            ],  # Fill remaining NaN with weighted average
            "valid_range": (0, 18250),  # Same range for remaining values
        },
    ],
    "BLDSUGAR": [
        {
            "step": "simple_replacement",
            "replacements": {888: 0},  # Don't know → No blood sugar checks
        },
        {
            "step": "redistribute",
            "values_to_redistribute": [777, 999],  # Don't know and Refused
            "valid_range": (
                0,
                40000,
            ),  # 0 to max ~99 times per day (36,135 per year) after transformation
        },
        {
            "step": "conditional_redistribute",
            "condition_column": "DIABETE3",
            "condition_value": 1,  # Has diabetes
            "values_to_redistribute": [
                np.nan
            ],  # Replace NaN values for diabetic patients
            "valid_range": (0, 40000),  # Distribution for diabetic patients
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "DIABETE3",
            "condition_value": lambda x: (x == 2)
            | (x == 3)
            | (x == 4),  # No diabetes, pre-diabetes, gestational only
            "target_values": [
                np.nan
            ],  # Replace remaining NaN values for non-diabetic patients
            "fill_value": 0,  # No blood sugar checks for non-diabetic patients
        },
    ],
    "FEETCHK2": [
        {
            "step": "simple_replacement",
            "replacements": {
                555: 0,
                888: 0,
            },  # Long time/Never and Don't know → No foot checks
        },
        {
            "step": "redistribute",
            "values_to_redistribute": [777, 999],  # Don't know and Refused
            "valid_range": (
                0,
                40000,
            ),  # 0 to max ~99 times per day (36,135 per year) after transformation
        },
        {
            "step": "conditional_redistribute",
            "condition_column": "DIABETE3",
            "condition_value": 1,  # Has diabetes
            "values_to_redistribute": [
                np.nan
            ],  # Replace NaN values for diabetic patients
            "valid_range": (0, 40000),  # Distribution for diabetic patients
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "DIABETE3",
            "condition_value": lambda x: (x == 2)
            | (x == 3)
            | (x == 4),  # No diabetes, pre-diabetes, gestational only
            "target_values": [
                np.nan
            ],  # Replace remaining NaN values for non-diabetic patients
            "fill_value": 0,  # No foot checks for non-diabetic patients
        },
    ],
    "FEETCHK": [
        {
            "step": "simple_replacement",
            "replacements": {88: 0},  # Don't know → No foot checks
        },
        {
            "step": "redistribute",
            "values_to_redistribute": [77, 99],  # Don't know and Refused
            "valid_range": (1, 76),  # Based on actual data range: 1-76
        },
        {
            "step": "conditional_redistribute",
            "condition_column": "DIABETE3",
            "condition_value": 1,  # Has diabetes
            "values_to_redistribute": [
                np.nan
            ],  # Replace NaN values for diabetic patients
            "valid_range": (1, 76),  # Distribution for diabetic patients
        },
        {
            "step": "conditional_fixed_fill",
            "condition_column": "DIABETE3",
            "condition_value": lambda x: (x == 2)
            | (x == 3)
            | (x == 4),  # No diabetes, pre-diabetes, gestational only
            "target_values": [
                np.nan
            ],  # Replace remaining NaN values for non-diabetic patients
            "fill_value": 0,  # No foot checks for non-diabetic patients
        },
    ],
}

# Random seed for reproducibility
RANDOM_SEED = 42
