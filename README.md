[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UcP9Py08)

# Machine Learning Project - Data Imputation and Analysis

## Overview
This project implements various data imputation techniques and machine learning models for analyzing and predicting outcomes in a healthcare dataset. It includes comprehensive data preprocessing, multiple imputation strategies, and model evaluation pipelines.

## Repository Structure
```
project-1-heartbreakers/
├── src/                    # Source code for core functionality
│   ├── implementations.py  # Core ML model implementations
│   ├── helpers.py         # Utility functions
│   ├── preprocessing.py   # Data preprocessing functions
│   ├── data_filling.py   # Data imputation implementations
│   ├── visualization.py   # Data visualization functions
│   └── mae_*.py          # MAE-specific implementations
├── notebooks/             # Jupyter notebooks for analysis
│   └── nan_analysis.ipynb # Missing value analysis notebook
├── configs/               # Configuration files
│   ├── imputation_configs.py  # Imputation parameters
│   └── mae_dictionnary.py    # MAE-specific configurations
├── scripts/              # Execution scripts
│   ├── run.py           # Main execution script
│   ├── apply_imputation.py    # Imputation pipeline
│   └── run_feature_removal.py # Feature selection script
├── data/                # Data directory
│   ├── raw/            # Original datasets
│   ├── processed/      # Processed datasets
│   └── features/       # Feature-related files
├── results/            # Model outputs and submissions
└── README.md
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/CS-433/project-1-heartbreakers.git
cd project-1-heartbreakers
```

2. Set up Python environment (Python 3.8+ recommended):
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Data Imputation Pipeline
To run the complete imputation pipeline:
```bash
python scripts/apply_imputation.py
```

### Feature Selection
To perform feature selection:
```bash
python scripts/run_feature_removal.py
```

### Model Training and Evaluation
To train and evaluate models:
```bash
python scripts/run.py
```

## Note on Naming Conventions
Some functions and files in this project may contain references to "mae" in their names (e.g., `mae_*` files or configurations). These names originated from one of the team member's initials (MAE) during initial development. While most instances have been updated to more descriptive names (e.g., `imputation_configs_cont.py`), you might still encounter some legacy naming patterns throughout the codebase. This naming inconsistency does not affect functionality.

## Project Components

### Data Preprocessing
- Missing value analysis and visualization
- Multiple imputation strategies including:
  - Mean imputation
  - Median imputation
  - Mode imputation
  - Custom imputation for specific features
- Feature engineering and selection

### Models
- Logistic Regression with regularization
- Cross-validation implementation
- F1-score optimization
- Threshold optimization for binary classification

### Evaluation
- Model performance metrics
- Cross-validation results
- Feature importance analysis
- Submission file generation

## Results
The results directory contains:
- Model submission files with different configurations
- Performance metrics across validation sets
- Feature selection and importance summaries

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Team
[Please add team member names and contact information]

## License
This project is part of the Machine Learning course (CS-433) at EPFL.

## Acknowledgments
- EPFL Machine Learning Course Staff
- Project supervisors and teaching assistants