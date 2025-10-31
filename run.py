import numpy as np
import os
from preprocessing import preprocessing_pipeline
from implementations import sigmoid, reg_logistic_regression_l1, reg_logistic_regression
from helpers import create_csv_submission
from implementations import oversampling

# OHE_idx_bis = [12]
#
# def get_indices_difference():
#     """
#     Returns the list of indices between 1 and 218 (inclusive) that are NOT in OHE_idx_bis.
#     """
#     all_indices = set(range(1, 219))
#     ohe_bis_set = set(OHE_idx_bis)
#     diff = sorted(list(all_indices - ohe_bis_set))
#     return diff
# normalization_idx_bis = get_indices_difference()


def run_logistic_regression_submission():
    # # Load the data
    # y_train_raw = np.genfromtxt(
    #     os.path.join('data', 'processed', 'dataset_features_removed', "y_train.csv"),
    #     delimiter=",",
    #     skip_header=1,
    #     dtype=int,
    #     usecols=1,
    # )
    # x_train_raw_ = np.genfromtxt(
    #     os.path.join('data', 'processed', 'dataset_features_removed', "x_train_filled.csv"), delimiter=",", skip_header=1
    # )
    # x_test_raw_ = np.genfromtxt(
    #     os.path.join('data', 'processed', 'dataset_features_removed', "x_test_filled.csv"), delimiter=",", skip_header=1
    # )
    # train_ids = x_train_raw_[:, 0].astype(dtype=int)
    # test_ids = x_test_raw_[:, 0].astype(dtype=int)
    #
    # # Save test_ids to submission_data/
    # os.makedirs('submission_data', exist_ok=True)
    # np.save(os.path.join('submission_data', 'test_ids.npy'), test_ids)

    # # Preprocess the data
    # x_train, x_test, normalized_cols_idx = preprocessing_pipeline(
    #     x_train_raw_, x_test_raw_, OHE_idx_bis, normalization_idx_bis, False
    # )
    # y_train = (y_train_raw + 1) / 2  # Converts -1 to 0, +1 to 1
    # tx_train = np.c_[np.ones((x_train.shape[0], 1)), x_train]
    # tx_test = np.c_[np.ones((x_test.shape[0], 1)), x_test]

    # # Best Hyperparameters
    # # Model: l2-regularized logistic regression
    # # Samplimg method: oversampling
    # lambda_ = 1.00e-06
    # gamma = 0.3
    # factor = 0.85
    # cutoff = 0.67
    # max_iters = 200
    # seed = 42

    # tx_train, y_train = oversampling(tx_train, y_train, factor)
    # initial_w = np.zeros(tx_train.shape[1])
    # w, _ = reg_logistic_regression(y_train, tx_train, lambda_, initial_w, max_iters, gamma)

    # # Predict on test set
    # z_test = tx_test @ w

    # # Save z_test to submission_data/
    # np.save(os.path.join('submission_data', 'z_test.npy'), z_test)

    test_ids = np.load("submission_data/test_ids.npy")
    z_test = np.load("submission_data/z_test.npy")
    cutoff = 0.67

    y_pred_prob = sigmoid(z_test)
    y_pred_binary = (y_pred_prob >= cutoff).astype(int)
    y_pred = 2 * y_pred_binary - 1  # Convert 0,1 back to -1,+1

    # Create submissions directory if it doesn't exist
    submissions_dir = "submissions"
    os.makedirs(submissions_dir, exist_ok=True)
    submission_name = f"best_submission.csv"
    submission_path = os.path.join(submissions_dir, submission_name)
    create_csv_submission(test_ids, y_pred, submission_path)
    print(f"\nSubmission saved as: {submission_path}")


if __name__ == "__main__":
    run_logistic_regression_submission()
