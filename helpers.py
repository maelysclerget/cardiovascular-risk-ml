"""Some helper functions for project 1."""

import csv
import numpy as np
import os
from itertools import combinations_with_replacement


def build_poly(x, idx, degree):
    """
    Polynomial feature expansion for selected features, concatenated to the original data.

    Parameters
    ----------
    x : (np.array) (n_samples, num_features) Input data.
    idx : (list or np.array) Indices of features to use for expansion.
    degree : (int) Maximum degree of polynomial features.

    Returns
    -------
    (np.array) (n_samples, num_features + num_poly_features)
    Original data with polynomial and cross terms of selected features up to the given degree appended as new columns.

    Examples
    --------
    >>> x = np.array([[1, 2, 7], [3, 4, 9]])
    >>> idx = np.array([0, 1])
    >>> degree = 2
    >>> build_poly(x, idx, degree)
    array([[ 1,  2,  1,  4,  7],
           [ 3,  4,  9, 16,  9]])
    """
    x_basis = x[:, idx]
    features = []

    for deg in range(1, degree + 1):
        for feature in idx:
            features.append(x_basis[:, feature] ** deg)

    poly_expansion = np.column_stack(features)
    x_rest = np.delete(x, idx, axis=1)
    return np.concatenate([poly_expansion, x_rest], axis=1)


def load_csv_data(data_path, sub_sample=False):
    """
    This function loads the data and returns the respectinve numpy arrays.
    Remember to put the 3 files in the same folder and to not change the names of the files.

    Args:
        data_path (str): datafolder path
        sub_sample (bool, optional): If True the data will be subsempled. Default to False.

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )

    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # sub-sample
    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids


def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})
