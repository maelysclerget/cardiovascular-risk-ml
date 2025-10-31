

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

## Implementation 🔌

Key methods and models:

- **Polynomial Basis Expansion:** Creating pseudo-polynomial feature expansion for normalized variables.
  ⚠️ Inter-feature products are not considered to avoid having thousands of features.
- **Oversampling/Undersampling:** Addressing class imbalance by resampling the training data.
- **L1 and L2 Logistic Regression:** Regularized logistic regression for binary classification.
- **Cross-Validation:** Robust model evaluation using k-fold cross-validation.

## Authors ✏️

- Maëlys Clerget <maelys.clerget@epfl.ch>
- Georges-Alex Nahas <georges-alex.nahas@epfl.ch>
- Aleksandar Mihaylov <aleksandar.mihaylov@epfl.ch>
