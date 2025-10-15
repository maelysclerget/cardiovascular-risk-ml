import numpy as np
import os
from helpers import load_csv_data, create_csv_submission
from implementations import logistic_regression, sigmoid
from preprocessing import preprocess_data


def main():
    """
    Main function to create a submission using logistic regression for classification.
    """
    # Load the data
    data_path = "dataset"
    x_train_raw, x_test_raw, y_train_raw, train_ids, test_ids = load_csv_data(data_path)
    
    # Preprocess the data (fill NaN values and remove highly correlated features)
    print("Preprocessing data...")
    x_train, x_test = preprocess_data(
        x_train_raw, x_test_raw, 
        fill_nan_values=True, 
        strategy='mean', 
        apply_correlation=True
    )
    
    # Normalize features to prevent numerical instability
    # Use training data statistics to normalize both train and test
    feature_means = np.mean(x_train, axis=0)
    feature_stds = np.std(x_train, axis=0)
    
    # Avoid division by zero for constant features
    feature_stds = np.where(feature_stds == 0, 1, feature_stds)
    
    x_train_normalized = (x_train - feature_means) / feature_stds
    x_test_normalized = (x_test - feature_means) / feature_stds
    
    print(f"Features normalized. Mean: {np.mean(x_train_normalized):.6f}, Std: {np.std(x_train_normalized):.6f}")
    
    # Convert labels from {-1, 1} to {0, 1} for logistic regression
    # -1 (no heart disease) → 0, +1 (heart disease) → 1
    y_train = (y_train_raw + 1) / 2  # Converts -1 to 0, +1 to 1
    
    # Add bias column (intercept) to the feature matrices
    tx_train = np.c_[np.ones((x_train_normalized.shape[0], 1)), x_train_normalized]
    tx_test = np.c_[np.ones((x_test_normalized.shape[0], 1)), x_test_normalized]
    
    # Initialize weights with smaller values to prevent overflow
    initial_w = np.zeros(tx_train.shape[1])  # Start with zeros for stability
    
    # Training parameters - much smaller learning rate for stability
    max_iters = 1000
    gamma = 0.001  # Reduced from 0.01 to prevent exploding gradients
    
    # Train the model using logistic regression
    print("Training logistic regression model...")
    print(f"Features after preprocessing: {x_train.shape[1]}")
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Max iterations: {max_iters}, Learning rate: {gamma}")
    
    w_optimal, loss = logistic_regression(y_train, tx_train, initial_w, max_iters, gamma)
    print(f"Training completed. Final loss: {loss:.6f}")
    
    # Make predictions on test set
    print("Making predictions on test set...")
    z = tx_test @ w_optimal
    y_pred_prob = sigmoid(z)  # Use numerically stable sigmoid function
    
    # Convert probabilities to binary predictions using 0.5 threshold
    # Then convert back to {-1, 1} format for submission
    y_pred_binary = (y_pred_prob >= 0.5).astype(int)  # 0 or 1
    y_pred = 2 * y_pred_binary - 1  # Convert 0,1 back to -1,+1
    
    print(f"Using probability threshold: 0.5")
    print(f"Probabilities >= 0.5 → heart disease (1)")
    print(f"Probabilities < 0.5 → no heart disease (-1)")
    
    # Create submissions directory if it doesn't exist
    submissions_dir = "submissions"
    os.makedirs(submissions_dir, exist_ok=True)
    
    # Create submission file in the submissions folder
    submission_name = "submission_logistic_regression.csv"
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
