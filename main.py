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
    x_train_raw = x_train_raw_[:, 1:]
    x_test_raw = x_test_raw_[:, 1:]

    # Preprocess the data (fill NaN, remove correlated features, and normalize)
    print("Preprocessing data...")
    x_train, x_test = preprocess_data(
        x_train_raw, x_test_raw, 
        fill_nan_values=False, 
        strategy='mean', 
        apply_correlation=False,
        normalize=True  # Enable normalization
    )
    
    # Convert labels from {-1, 1} to {0, 1} for logistic regression
    # -1 (no heart disease) → 0, +1 (heart disease) → 1
    y_train = (y_train_raw + 1) / 2  # Converts -1 to 0, +1 to 1
    
    # Add bias column (intercept) to the feature matrices
    tx_train = np.c_[np.ones((x_train.shape[0], 1)), x_train]
    tx_test = np.c_[np.ones((x_test.shape[0], 1)), x_test]

    
    # PARAMETERS FOR BALANCED CROSS-VALIDATION
    max_iters = 500              # Keep good convergence
    # Reduced parameter grid
    lambdas = np.logspace(-3, 1, 8)   # 8 values from 0.001 to 10
    gammas = np.logspace(-4, -1, 6)   # 6 values from 0.0001 to 0.1
    k_fold = 10                  # Keep robust validation
    
    # Reduced number of initializations
    w_inits = [
        np.zeros(tx_train.shape[1]),  # Zero initialization (standard)
        np.random.normal(0, 0.01, tx_train.shape[1]),  # Small random normal
        np.ones(tx_train.shape[1]) * 0.01,  # Small constant
    ]
    
    # Train the model using logistic regression with multiple initializations
    print("Training logistic regression model with EXTENSIVE CV settings...")
    print(f"Number of different initializations: {len(w_inits)}")
    print(f"Features after preprocessing: {x_train.shape[1]}")
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Max iterations: {max_iters}, K-folds: {k_fold}")
    print(f"Parameter combinations: {len(lambdas)} × {len(gammas)} = {len(lambdas)*len(gammas)}")
    print(f"Total CV runs: {len(lambdas)*len(gammas)*k_fold} = {len(lambdas)*len(gammas)*k_fold}")
    
    # Try each initialization
    best_overall_lambda = None
    best_overall_gamma = None
    best_overall_loss = float('inf')
    best_overall_w = None
    
    for init_idx, w_init in enumerate(w_inits):
        print(f"\nTrying initialization {init_idx + 1}/{len(w_inits)}...")
        best_lambda, best_gamma, best_loss, results = logistic_cross_validation_demo(
            y_train, tx_train, k_fold, lambdas, gammas, max_iters, initial_w=w_init, seed=42+init_idx
        )
        print(f"Init {init_idx + 1} completed. Loss: {best_loss:.6f}")
        
        if best_loss < best_overall_loss:
            best_overall_loss = best_loss
            best_overall_lambda = best_lambda
            best_overall_gamma = best_gamma
            best_overall_w = w_init
            print(f"New best initialization found!")
    
    print(f"\nBest overall results:")
    print(f"Lambda: {best_overall_lambda:.6f}")
    print(f"Gamma: {best_overall_gamma:.6f}")
    print(f"Loss: {best_overall_loss:.6f}")
    
    # Make predictions on test set using best parameters
    print("\nMaking predictions on test set...")
    w_optimal, loss = reg_logistic_regression(
        y_train, tx_train, 
        best_overall_lambda, 
        best_overall_w, 
        max_iters, 
        best_overall_gamma, 
        verbose=False
    )
    
    # Get probabilities for training set to optimize threshold for F1
    print("Optimizing threshold for F1 score...")
    z_train = tx_train @ w_optimal
    y_train_prob = sigmoid(z_train)
    
    # Test different thresholds to maximize F1 on training set
    thresholds = np.arange(0.25, 0.76, 0.01)  # More fine-grained threshold search
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
    submissions_dir = 'submissions'
    os.makedirs(submissions_dir, exist_ok=True)
    
    # Create submission file in the results/submissions folder
    submission_name = f"submission_reg_logistic_f1_optimized_th{best_threshold:.3f}_extCV.csv"
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
