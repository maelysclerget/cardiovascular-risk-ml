import numpy as np
import os
from preprocessing import preprocessing_pipeline

OHE_idx =  [1, 2, 3, 4, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 59, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 102, 103, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 152, 153, 154, 155, 156, 157, 158, 159, 167, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 182, 187, 188, 190, 192, 201, 202, 207, 208, 214, 215, 216, 217, 218]
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
    print(f"Number of OHE features: {len(OHE_idx)}")
    print(f"Number of features to normalize: {len(normalization_idx)}")
    print(f"Total features before OHE: {len(OHE_idx) + len(normalization_idx)}")

    # Preprocess the data (fill NaN, remove correlated features, and normalize)
    print("\nPreprocessing data...")
    x_train, x_test, normalized_cols_idx = preprocessing_pipeline(
        x_train_raw_, x_test_raw_, OHE_idx, normalization_idx, True, 0.9
    )

    #  x_train = x_train_pre[:, 1:]  # get rid of ID column, already removed in preprocessing_pipeline
    # x_test = x_test_pre[:, 1:]  # get rid of ID column, already removed in preprocessing_pipeline

    print("\nShapes after preprocessing:")
    print("Shape of x_train after preprocessing:", x_train.shape)
    print("Shape of x_test after preprocessing:", x_test.shape)
    
    # Verify train and test have same number of features
    if x_train.shape[1] != x_test.shape[1]:
        print("\nWARNING: Train and test sets have different numbers of features!")
    
    # Check the range of values in the encoded columns
    # We expect 0s and 1s in the one-hot encoded columns
    encoded_range_train = np.unique(x_train[:, np.arange(len(normalization_idx), x_train.shape[1])])
    encoded_range_test = np.unique(x_test[:, np.arange(len(normalization_idx), x_test.shape[1])])
    print("\nOne-hot encoding validation:")
    print(f"Unique values in train encoded columns: {encoded_range_train}")
    print(f"Unique values in test encoded columns: {encoded_range_test}")
    
    # Calculate the number of new columns created by OHE
    original_features = len(OHE_idx) + len(normalization_idx)
    final_features = x_train.shape[1]
    new_columns = final_features - original_features
    print(f"\nOne-Hot Encoding impact:")
    print(f"Original feature count: {original_features}")
    print(f"Final feature count: {final_features}")
    print(f"New columns created by OHE: {new_columns}")

    # # Convert labels from {-1, 1} to {0, 1} for logistic regression
    # y_train = (y_train_raw + 1) / 2  # Converts -1 to 0, +1 to 1
    
    # # Add bias column (intercept) to the feature matrices
    # tx_train = np.c_[np.ones((x_train.shape[0], 1)), x_train]
    # tx_test = np.c_[np.ones((x_test.shape[0], 1)), x_test]

    # # Use best hyperparameters from previous run
    # lambda_ = 0.001
    # gamma = 0.1
    # max_iters = 500
    # initial_w = np.zeros(tx_train.shape[1])

    # print("\nTraining logistic regression with best hyperparameters:")
    # print(f"Lambda: {lambda_}")
    # print(f"Gamma: {gamma}")
    # print(f"Max iterations: {max_iters}")

    # # Train the model using logistic regression
    # from implementations import reg_logistic_regression, sigmoid, precision_recall_f1
    # w_optimal, loss = reg_logistic_regression(
    #     y_train, tx_train, 
    #     lambda_, 
    #     initial_w, 
    #     max_iters, 
    #     gamma, 
    #     verbose=False
    # )

    # # Get probabilities for training set to optimize threshold for F1
    # print("\nOptimizing threshold for F1 score...")
    # z_train = tx_train @ w_optimal
    # y_train_prob = sigmoid(z_train)
    
    # # Test different thresholds to maximize F1 on training set
    # thresholds = np.arange(0.25, 0.76, 0.01)  # Fine-grained threshold search
    # best_f1 = 0
    # best_threshold = 0.5
    
    # for threshold in thresholds:
    #     y_pred_train_binary = (y_train_prob >= threshold).astype(int)
    #     # Use the precision_recall_f1 function
    #     precision, recall, f1 = precision_recall_f1(y_train, y_pred_train_binary)
            
    #     if f1 > best_f1:
    #         best_f1 = f1
    #         best_threshold = threshold
    
    # print(f"Best threshold for F1: {best_threshold:.3f} (F1 = {best_f1:.4f})")
    
    # # Apply best threshold to test set
    # z_test = tx_test @ w_optimal
    # y_pred_prob = sigmoid(z_test)
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
    # submission_name = f"submission_reg_logistic_lambda{lambda_}_gamma{gamma}_th{best_threshold:.3f}.csv"
    # submission_path = os.path.join(submissions_dir, submission_name)
    # create_csv_submission(test_ids, y_pred, submission_path)
    # print(f"\nSubmission saved as: {submission_path}")

if __name__ == "__main__":
    main()