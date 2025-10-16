import numpy as np
import os
from helpers import load_csv_data, create_csv_submission
from implementations import logistic_regression, sigmoid, logistic_cross_validation_demo, reg_logistic_regression
from preprocessing import preprocess_data


def main():
    """
    Main function to create a submission using logistic regression for classification.
    """
    # Load the data
    data_path = "dataset"
    x_train_raw, x_test_raw, y_train_raw, train_ids, test_ids = load_csv_data(data_path)
    
    # Preprocess the data (fill NaN, remove correlated features, and normalize)
    print("Preprocessing data...")
    x_train, x_test = preprocess_data(
        x_train_raw, x_test_raw, 
        fill_nan_values=True, 
        strategy='mean', 
        apply_correlation=True,
        normalize=True  # Enable normalization
    )
    
    # Convert labels from {-1, 1} to {0, 1} for logistic regression
    # -1 (no heart disease) → 0, +1 (heart disease) → 1
    y_train = (y_train_raw + 1) / 2  # Converts -1 to 0, +1 to 1
    
    # Add bias column (intercept) to the feature matrices
    tx_train = np.c_[np.ones((x_train.shape[0], 1)), x_train]
    tx_test = np.c_[np.ones((x_test.shape[0], 1)), x_test]

    
    # OPTIMIZED PARAMETERS FOR SPEED
    max_iters = 300              # Reduced from 1000 (3x faster)
    lambdas = np.array([0.1, 1.0])     # Reduced from 4 to 2 values (2x faster)
    gammas = np.array([0.001, 0.01])   # Reduced from 3 to 2 values (1.5x faster)
    k_fold = 5                   # Reduced from 10 to 5 folds (2x faster)
    
    # Train the model using logistic regression
    print("Training logistic regression model with FAST CV settings...")
    print(f"Features after preprocessing: {x_train.shape[1]}")
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Max iterations: {max_iters}, K-folds: {k_fold}")
    print(f"Parameter combinations: {len(lambdas)} × {len(gammas)} = {len(lambdas)*len(gammas)}")
    print(f"Total CV runs: {len(lambdas)*len(gammas)*k_fold} = {len(lambdas)*len(gammas)*k_fold}")
    
    best_lambda, best_gamma, best_loss, results = logistic_cross_validation_demo(y_train, tx_train, k_fold, lambdas, gammas, max_iters)
    print(f"Training completed. Final loss: {best_loss:.6f}")
    
    # Make predictions on test set
    print("Making predictions on test set...")
    w_initial = np.zeros(tx_train.shape[1])  # Initialize weights for final training
    w_optimal, loss = reg_logistic_regression(y_train, tx_train, best_lambda, w_initial, max_iters, best_gamma, verbose=False)
    
    # Get probabilities for training set to optimize threshold for F1
    print("Optimizing threshold for F1 score...")
    z_train = tx_train @ w_optimal
    y_train_prob = sigmoid(z_train)
    
    # Test different thresholds to maximize F1 on training set
    thresholds = np.arange(0.1, 0.9, 0.05)  # Test thresholds from 0.1 to 0.85
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred_train_binary = (y_train_prob >= threshold).astype(int)
        
        # Calculate F1 score
        tp = np.sum((y_pred_train_binary == 1) & (y_train == 1))
        fp = np.sum((y_pred_train_binary == 1) & (y_train == 0))
        fn = np.sum((y_pred_train_binary == 0) & (y_train == 1))
        
        if tp + fp > 0:  # Avoid division by zero
            precision = tp / (tp + fp)
        else:
            precision = 0
            
        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0
            
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
            
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"Best threshold for F1: {best_threshold:.3f} (F1 = {best_f1:.4f})")
    
    # Apply best threshold to test set
    z_test = tx_test @ w_optimal
    y_pred_prob = sigmoid(z_test)
    y_pred_binary = (y_pred_prob >= best_threshold).astype(int)  # 0 or 1
    y_pred = 2 * y_pred_binary - 1  # Convert 0,1 back to -1,+1
    
    print(f"Using optimized threshold: {best_threshold:.3f}")
    print(f"Probabilities >= {best_threshold:.3f} → heart disease (1)")
    print(f"Probabilities < {best_threshold:.3f} → no heart disease (-1)")
    
    # Create submissions directory if it doesn't exist
    submissions_dir = "submissions"
    os.makedirs(submissions_dir, exist_ok=True)
    
    # Create submission file in the submissions folder
    submission_name = f"submission_reg_logistic_f1_optimized_th{best_threshold:.3f}_10cv.csv"
    submission_path = os.path.join(submissions_dir, submission_name)
    create_csv_submission(test_ids, y_pred, submission_path)
    print(f"Submission saved as: {submission_path}")
    
    # Print some statistics
    print(f"\nPrediction statistics:")
    print(f"Number of positive predictions (1): {np.sum(y_pred == 1)}")
    print(f"Number of negative predictions (-1): {np.sum(y_pred == -1)}")
    print(f"Percentage positive: {100 * np.sum(y_pred == 1) / len(y_pred):.2f}%")
    print(f"Average predicted probability: {np.mean(y_pred_prob):.4f}")
    print(f"Probability range: [{np.min(y_pred_prob):.4f}, {np.max(y_pred_prob):.4f}]")


if __name__ == "__main__":
    main()
