

# Predicting Cardiovascular Disease with Machine Learning ❤️‍🩹

## Project Description 🚩

Cardiovascular diseases (CVDs) are a leading cause of death worldwide. This project leverages machine learning to predict the risk of developing CVDs, such as heart attacks, using health, lifestyle, and demographic data. We use data from the Behavioral Risk Factor Surveillance System (BRFSS) to build binary classification models that estimate the likelihood of Myocardial Infarct or Coronary Heart Disease (MICHD) for an individual.

## Table of Contents 📋

- [Project Description](#project-description)
- [UV Virtual Environment Setup](#uv-virtual-environment-setup)
- [Data Processing](#data-processing)
- [Implementation](#implementation)
- [Authors](#authors)

## UV Virtual Environment Setup 🌳

We use [uv](https://github.com/astral-sh/uv) as the project's reproducible Python virtual environment. To set up the environment,
execute the following commands after cloning the project:

1. **Install uv:** (if not already installed):
   ```bash
   pip install uv
   ```

2. **Ensure existence of a viratual environment:**
   ```bash
   uv venv .venv
   ```

3. **Activate the environment:**
   ```bash
   source .venv/bin/activate
   ```

4. **Install project dependencies:**
   ```bash
   uv pip install .
   ```

## Data Processing 📊

Our data pipeline includes:

- **Imputation:** Missing values were carefully imputated separately for each single feature to ensure data quality. The impuation was based on the weighted distribution created by the existing answers to each question. Simple weighted average was not employed!
- **Feature removal:** Certain features were removed as they contained too many missing values or did not apply to the entire patient population
- **One-hot encoding:** Converting categorical variables into binary indicator columns. 
  ⚠️ Dummy Variable Trap avoided: (k - 1) One-hot encoding
- **Normalization:** Normalizing ordinal and continuous features to have zero mean and unit variance.
- **Correlated Feature Analysis:** Optional removal of highly correlated, normalized features. 
  ⚠️ Does not apply to One-hot encoded features!

### Beware: Data Filling Procedure 🚨

We imputated the various features one-by-one manually. Our imputation process can be found in the following files:

| File                                      | Purpose                                                                                   |
|:------------------------------------------ |:------------------------------------------------------------------------------------------|
| configs/imputation_configs_OHE.py          | Imputation strategies for categorical (OHE) features: value replacement, conditional, weighted, and complex imputation. |
| configs/imputation_configs_cont.py         | Imputation configs for continuous features: rules for value replacement and redistribution. |
| scripts/apply_imputation.py                | Applies all OHE imputation configs to train/test data using multi-step imputation.         |
| scripts/apply_cont_features_imputation.py  | Imputes a single continuous feature using detailed configs and custom logic.               |
| scripts/ordered_imputation.py              | Runs ordered imputation pipeline, handling feature dependencies in a specific sequence.    |
| scripts/remove_features_from_filled.py     | Removes specified columns from a filled CSV, preserving format.                            |
| scripts/run_feature_removal.py             | Removes a predefined list of features from the dataset using the above module.             |
| src/cont_features_particular_cases.py        | Implements custom logic and helpers for reading CSVs and handling particular feature cases. |
| src/cont_features_data_filling.py            | Advanced imputation and data filling for continuous features, with specialized techniques.  |
| src/cont_feature_application.py              | Main pipeline for continuous feature imputation, supporting multiple strategies and stats.   |

Thus, the filled database which we generated and used for our group's submissions is different from the raw database from the BRFSS. Due to the large size of the database, we did not include it in the Git Repository.

If you need access to our filled database to review our project, please email us so that we send it using WeTransfer, for example.

Finally, you will not be able to run our data preprocessing pipeline described above unless you ask us to send you our filled database.

## Implementation 🔌

Key methods and models:

- **Polynomial Basis Expansion:** Creating pseudo-polynomial feature expansion for normalized variables.
  ⚠️ Inter-feature products are not considered to avoid having thousands of features.
- **Oversampling/Undersampling:** Addressing class imbalance by resampling the training data.
- **L1 and L2 Logistic Regression:** Regularized logistic regression for binary classification.
- **Cross-Validation:** Robust model evaluation using k-fold cross-validation.

## Best Model and Hyperparameters 🛠️

Please find below the model and the associated hyperparameters that generated our best submission.

| Hyperparameter  | Value                  |
|:--------------- |:---------------------- |
| Algorithm       | l2 Logistic Regression |
| Lambda          | 1.00e-06               |
| Gamma           | 3.00e-01               |
| Degree          | 1                      |
| Sampling        | Oversampling           |
| Sampling Factor | 0.85                   |
| Cutoff          | 0.67                   |

### Beware: Reproducing our Best Results 🚨

In order to reproduce our best results, we have provided the **test_id** and the **z_test** in a separate folder called **submission_data/**. The code to then recreate it is found in **run.py**. 

As described above, due to the manual and extensive imputation of the dataset we cannot provide you simply with our optimal weights or the code to generate them. However, if you request our filled database, you can uncomment all line in **run.py** and recreate our best submission without simply opening the provided **test_id** and **z_test** from **submission_data/**.

## Authors ✏️

- Maëlys Clerget <maelys.clerget@epfl.ch>
- Georges-Alex Nahas <georges-alex.nahas@epfl.ch>
- Aleksandar Mihaylov <aleksandar.mihaylov@epfl.ch>
