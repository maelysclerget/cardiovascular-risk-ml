import numpy as np
import os
from preprocessing import preprocessing_pipeline
from visualization import plot_metrics_vs_hyperparameter, plot_metric
import matplotlib.pyplot as plt
from collections import defaultdict

OHE_idx =  [1, 2, 3, 4, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 59, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 102, 103, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 152, 153, 154, 155, 156, 157, 158, 159, 167, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 182, 187, 188, 190, 192, 201, 202, 207, 208, 214, 215, 216, 217, 218]
#OHE_idx_bis = [12,14,18,21,22,24,25,26,27,28,29,31,36,37,41,42,43,44,45,46,47,48,49,59,62,63,67,71,76,85,93,100,102,103,114,126,169,173,180,188]
OHE_idx_bis = [12]

def get_indices_difference():
    """
    Returns the list of indices between 1 and 218 (inclusive) that are NOT in OHE_idx_bis.
    """
    all_indices = set(range(1, 219))
    ohe_bis_set = set(OHE_idx_bis)
    diff = sorted(list(all_indices - ohe_bis_set))
    return diff

normalization_idx_bis = get_indices_difference()
normalization_idx = [5, 6, 7, 9, 10, 11, 35, 39, 55, 56, 57, 58, 65, 77, 78, 79, 80, 81, 101, 104, 105, 106, 107, 125, 151, 160, 161, 162, 163, 164, 165, 166, 168, 181, 183, 184, 185, 186, 189, 191, 193, 194, 195, 196, 197, 198, 199, 200, 203, 204, 205, 206, 209, 210, 211, 212, 213]

def main():
    """
    Main function to create a submission using logistic regression for classification.
    """
    # Load the data
    y_train_raw = np.genfromtxt(
        os.path.join('data', 'processed', 'dataset_features_removed', "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    
    x_train_raw_ = np.genfromtxt(
        os.path.join('data', 'processed', 'dataset_features_removed', "x_train_filled.csv"), delimiter=",", skip_header=1
    )
    x_test_raw_ = np.genfromtxt(
        os.path.join('data', 'processed', 'dataset_features_removed', "x_test_filled.csv"), delimiter=",", skip_header=1
    )

    train_ids = x_train_raw_[:, 0].astype(dtype=int)
    test_ids = x_test_raw_[:, 0].astype(dtype=int)

    # Print shapes before preprocessing
    print("\nShapes before preprocessing:")
    print("x_train_raw shape:", x_train_raw_.shape)
    print("x_test_raw shape:", x_test_raw_.shape)
    print(f"Number of OHE features: {len(OHE_idx_bis)}")
    print(f"Number of features to normalize: {len(normalization_idx_bis)}")
    print(f"Total features before OHE: {len(OHE_idx_bis) + len(normalization_idx_bis)}")

    # Preprocess the data (fill NaN, remove correlated features, and normalize)
    print("\nPreprocessing data...")
    x_train, x_test, normalized_cols_idx = preprocessing_pipeline(
        x_train_raw_, x_test_raw_, OHE_idx_bis, normalization_idx_bis, False
    )
    
    print("\nShapes after preprocessing:")
    print("Shape of x_train after preprocessing:", x_train.shape)
    print("Shape of x_test after preprocessing:", x_test.shape)
    
    # Verify train and test have same number of features
    if x_train.shape[1] != x_test.shape[1]:
        print("\nWARNING: Train and test sets have different numbers of features!")
    
    # Check the range of values in the encoded columns
    # We expect 0s and 1s in the one-hot encoded columns
    encoded_range_train = np.unique(x_train[:, np.arange(len(normalization_idx_bis), x_train.shape[1])])
    encoded_range_test = np.unique(x_test[:, np.arange(len(normalization_idx_bis), x_test.shape[1])])
    print("\nOne-hot encoding validation:")
    print(f"Unique values in train encoded columns: {encoded_range_train}")
    print(f"Unique values in test encoded columns: {encoded_range_test}")

    # Convert labels from {-1, 1} to {0, 1} for logistic regression
    y_train = (y_train_raw + 1) / 2  # Converts -1 to 0, +1 to 1
    
    # Add bias column (intercept) to the feature matrices
    tx_train = np.c_[np.ones((x_train.shape[0], 1)), x_train]
    tx_test = np.c_[np.ones((x_test.shape[0], 1)), x_test]
    
    # Hyperparameter tuning (example values, adjust as needed)
    lambda_list    = [1e-4, 1e-5]  # Regularization strength (4 values)
    gamma_list     = [0.15,0.20,0.25,0.30]   # Learning rate (4 values)
    degree_list    = [1]                                  # Polynomial degree
    sampling_list  = [None]      # Sampling strategy
    factor_list    = [1]                                # Sampling factor
    algorithm_list = ['l2']                           # Regularization type
    cutoff_list    = [0.67]                   # Classification threshold
    
    # Perform cross-validation to find the best hyperparameters
    print("\nStarting cross-validation for hyperparameter tuning...")
    from implementations import logistic_cross_validation_demo
    best_f1_result, best_loss_result, all_results = logistic_cross_validation_demo(
        y_train,
        tx_train,
        k_fold=5,
        normalized_cols_idx=normalized_cols_idx,
        degrees=degree_list,
        lambdas=lambda_list,
        gammas=gamma_list,
        samplings=sampling_list,
        factors=factor_list,
        algorithms=algorithm_list,
        cutoffs=cutoff_list,
        max_iters=200,
        seed=42,
        verbose=True,
    )
    print("\nBest hyperparameters by F1 Score:")
    print(best_f1_result)
    print("\nBest hyperparameters by Loss:")
    print(best_loss_result) 

    # Visualize results
    plot_metrics_vs_hyperparameter(all_results, 'lambda', 'avg_f1_te')
    plot_metrics_vs_hyperparameter(all_results, 'gamma', 'avg_f1_te')
    plot_metrics_vs_hyperparameter(all_results, 'cutoff', 'avg_f1_te')
    plot_metrics_vs_hyperparameter(all_results, 'factor', 'avg_f1_te')
    plot_metric(all_results, 'avg_f1_te')   
    plot_metric(all_results, 'avg_loss_te') 

    # f1_scores = [r['avg_f1_te'] for r in all_results]
    # lambdas   = [r['lambda'] for r in all_results]

    # data = defaultdict(list)
    # for lam, f1 in zip(lambdas, f1_scores):
    #     data[lam].append(f1)

    # means = {lam: np.mean(vals) for lam, vals in data.items()}
    # stds  = {lam: np.std(vals)  for lam, vals in data.items()}

    # plt.figure(figsize=(8, 5))
    # plt.errorbar(list(means.keys()), list(means.values()), yerr=list(stds.values()),
    #             fmt='o-', capsize=5)
    # plt.xlabel('Lambda')
    # plt.ylabel('Mean F1 Score')
    # plt.title('Mean ± Std F1 Score vs Lambda')
    # plt.grid(True)
    # plt.savefig('f1_score_vs_lambda_3.png')

    # # Plotting mean F1 Score per Gamma value
    # gammas   = [r['gamma'] for r in all_results]

    # data = defaultdict(list)
    # for gamma, f1 in zip(gammas, f1_scores):
    #     data[gamma].append(f1)

    # means = {gamma: np.mean(vals) for gamma, vals in data.items()}
    # stds  = {gamma: np.std(vals)  for gamma, vals in data.items()}

    # plt.figure(figsize=(8, 5))
    # plt.errorbar(list(means.keys()), list(means.values()), yerr=list(stds.values()),
    #             fmt='o-', capsize=5)
    # plt.xlabel('Gamma')
    # plt.ylabel('Mean F1 Score')
    # plt.title('Mean ± Std F1 Score vs Gamma')
    # plt.grid(True)
    # plt.savefig('f1_score_vs_gamma_3.png')

    # # Plotting mean F1 Score per Threshold value
    # cutoffs = [r['cutoff'] for r in all_results]

    # data = defaultdict(list)
    # for cutoff, f1 in zip(cutoffs, f1_scores):
    #     data[cutoff].append(f1)

    # means = {cutoff: np.mean(vals) for cutoff, vals in data.items()}
    # stds  = {cutoff: np.std(vals)  for cutoff, vals in data.items()}

    # plt.figure(figsize=(8, 5))
    # plt.errorbar(list(means.keys()), list(means.values()), yerr=list(stds.values()),
    #             fmt='o-', capsize=5)
    # plt.xlabel('Threshold (Cutoff)')
    # plt.ylabel('Mean F1 Score')
    # plt.title('Mean ± Std F1 Score vs Threshold')
    # plt.grid(True)
    # plt.savefig('f1_score_vs_threshold_3.png')

    # # Plotting mean F1 Score per Sampling Factor value
    # factors = [r['factor'] for r in all_results]

    # data = defaultdict(list)
    # for factor, f1 in zip(factors, f1_scores):
    #     data[factor].append(f1)

    # means = {factor: np.mean(vals) for factor, vals in data.items()}
    # stds  = {factor: np.std(vals)  for factor, vals in data.items()}

    # plt.figure(figsize=(8, 5))
    # plt.errorbar(list(means.keys()), list(means.values()), yerr=list(stds.values()),
    #             fmt='o-', capsize=5)
    # plt.xlabel('Sampling Factor')
    # plt.ylabel('Mean F1 Score')
    # plt.title('Mean ± Std F1 Score vs Sampling Factor')
    # plt.grid(True)
    # plt.savefig('f1_score_vs_sampling_factor_3.png')

    # metric_array = np.array([res['avg_f1_te'] for res in all_results])

    # plt.figure(figsize=(10, 6))
    # plt.plot(metric_array, marker='o')
    # plt.xlabel('Hyperparameter Combination Index')
    # plt.ylabel('F1 Score')
    # plt.title(f'F1 Score over Hyperparameter Combination Index')
    # plt.grid(True)
    # plt.savefig('f1_score_over_hyperparameter_index_3.png')
    # plt.show()
    
    # ## Train final model on full training data with best hyperparameters
    # z_test = tx_test @ best_f1_result["weights"]
    # from implementations import sigmoid
    # y_pred_prob = sigmoid(z_test)
    # best_threshold = best_f1_result["cutoff"]
    # y_pred_binary = (y_pred_prob >= best_threshold).astype(int)  # 0 or 1
    # y_pred = 2 * y_pred_binary - 1  # Convert 0,1 back to -1,+1
    
    # print(f"\nPrediction statistics:")
    # print(f"Number of positive predictions (1): {np.sum(y_pred == 1)}")
    # print(f"Number of negative predictions (-1): {np.sum(y_pred == -1)}")
    # print(f"Percentage positive: {100 * np.sum(y_pred == 1) / len(y_pred):.2f}%")
    # print(f"Average predicted probability: {np.mean(y_pred_prob):.4f}")
    # print(f"Probability range: [{np.min(y_pred_prob):.4f}, {np.max(y_pred_prob):.4f}]")
    
    # # Create submissions directory if it doesn't exist
    # submissions_dir = 'submissions'
    # os.makedirs(submissions_dir, exist_ok=True)
    
    # # Create submission file
    # from helpers import create_csv_submission
    # submission_name = f"submission_reg_logistic.csv"
    # submission_path = os.path.join(submissions_dir, submission_name)
    # create_csv_submission(test_ids, y_pred, submission_path)
    # print(f"\nSubmission saved as: {submission_path}")

if __name__ == "__main__":
    main()