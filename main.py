import numpy as np
import os
from helpers import load_csv_data, create_csv_submission
from implementations import logistic_regression, sigmoid, logistic_cross_validation_demo, reg_logistic_regression
from preprocessing import preprocess_data


def main():
    """
    Main function to create a submission using logistic regression for classification.
    """
    