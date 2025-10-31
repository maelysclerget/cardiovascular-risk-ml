import numpy as np
from helpers import build_poly
from preprocessing import oversampling, undersampling
from itertools import product


def mean_squared_error_loss(y, tx, w):
    """
    Compute the loss by Mean Squared Error (MSE).

    Parameters
    ----------
    y : (np.array) Output data points
    tx : (np.array) Input data points
    w : (np.array) Weights

    Returns
    -------
    (float) MSE loss
    """
    n = tx.shape[0]  # Number of samples
    e = y - tx @ w
    return (e.T @ e) / (2 * n)


def compute_gradient_MSE(y, tx, w):
    """
    Compute the gradient by Mean Squared Error (MSE).

    Parameters
    ----------
    y : (np.array) Output data points
    tx : (np.array) Input data points
    w : (np.array) Weights

    Returns
    -------
    (np.array) Gradient of MSE loss
    """
    n = tx.shape[0]  # Number of samples
    e = y - tx @ w
    return -(tx.T @ e) / n


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Gradient Descent (GD) algorithm for minimizing the MSE loss.

    Parameters
    ----------
    y : (np.array) Output data points
    tx : (np.array) Input data points
    initial_w : (np.array) Initial weights
    max_iters : (int) Maximal number of iterations
    gamma : (float) Learning rate

    Returns
    -------
    (np.array, float) Final weights and their corresponding loss
    """
    w = initial_w

    for _ in range(max_iters):
        gradient = compute_gradient_MSE(y, tx, w)
        w = w - gamma * gradient

    loss = mean_squared_error_loss(y, tx, w)

    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Stochastic Gradient Descent (SGD) algorithm for minimizing the MSE loss.

    Parameters
    ----------
    y : (np.array) Output data points
    tx : (np.array) Input data points
    initial_w : (np.array) Initial weights
    max_iters : (int) Maximal number of iterations
    gamma : (float) Learning rate

    Returns
    -------
    (np.array, float) Final weights and their corresponding loss
    """
    w = initial_w
    n = tx.shape[0]  # Number of samples

    for _ in range(max_iters):
        ind = np.random.randint(
            low=0, high=n
        )  # Randomly selected index/data point for gradient and loss computation
        gradient = compute_gradient_MSE(y[ind : ind + 1], tx[ind : ind + 1], w)
        w = w - gamma * gradient

    loss = mean_squared_error_loss(y[ind : ind + 1], tx[ind : ind + 1], w)

    return w, loss


def least_squares(y, tx):
    """
    Least squares regression using normal equations.

    Parameters
    ----------
    y : (np.array) Output data points
    tx : (np.array) Input data points

    Returns
    -------
    (np.array, float) Optimal weights and their corresponding MSE loss
    """
    # w = (X^T X)^(-1) X^T y
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)

    # Compute the MSE loss for the optimal weights
    loss = mean_squared_error_loss(y, tx, w)

    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations with L2 regularization.

    Parameters
    ----------
    y : (np.array) Output data points
    tx : (np.array) Input data points
    lambda_ : (float) Regularization parameter

    Returns
    -------
    (np.array, float) Optimal weights and their corresponding MSE loss
    """
    # w = (X^T X + λI)^(-1) X^T y
    d = tx.shape[1]  # Number of features
    n = tx.shape[0]  # Number of samples
    identity = np.identity(d)
    w = np.linalg.solve(tx.T @ tx + 2 * n * lambda_ * identity, tx.T @ y)

    return w, mean_squared_error_loss(y, tx, w)


def sigmoid(eta):
    """
    Numerically stable sigmoid function.

    Parameters
    ----------
    eta : (np.array or float)
        Linear predictor values (can be scalar, vector, or matrix).

    Returns
    -------
    (np.array or float) The logistic sigmoid applied elementwise to `eta`.
    """
    # Clip eta to prevent overflow
    eta = np.clip(eta, -500, 500)

    # Use numerically stable computation
    # For eta > 0: sigmoid(eta) = 1 / (1 + exp(-eta))
    # For eta <= 0: sigmoid(eta) = exp(eta) / (1 + exp(eta))
    pos_mask = eta > 0
    result = np.zeros_like(eta)

    # Positive values
    result[pos_mask] = 1 / (1 + np.exp(-eta[pos_mask]))

    # Negative values
    exp_eta = np.exp(eta[~pos_mask])
    result[~pos_mask] = exp_eta / (1 + exp_eta)

    return result


def logistic_loss_function(y, tx, w, lambda_=0):
    """
    Logistic regression loss function with optional L2 regularization.

    Parameters
    ----------
    y : (np.array) Output data points
    tx : (np.array) Input data points
    w : (np.array) Weights
    lambda_ : (float) Regularization parameter

    Returns
    -------
    (float) Logistic regression loss with optional L2 penalty:
    """
    n = tx.shape[0]  # Number of samples
    z = tx @ w
    # loss = np.sum(y * np.log(sigmoid(tx @ w)) + (1 - y) * np.log(1 - sigmoid(tx @ w))) / -n + lambda_ * np.sum(w*w)
    loss = ((-y.T @ z) + np.sum(np.log(1 + np.exp(z)))) / n + lambda_ * np.sum(w * w)
    return loss


def logisitc_loss_function_l1(y, tx, w, lambda_=0):
    """
    Logistic regression loss function with L1 regularization.

    Parameters
    ----------
    y : (np.array) Output data points
    tx : (np.array) Input data points
    w : (np.array) Weights
    lambda_ : (float) Regularization parameter

    Returns
    -------
    (float) Logistic regression loss with L1 penalty:
    """
    n = tx.shape[0]  # Number of samples
    z = tx @ w
    loss = ((-y.T @ z) + np.sum(np.log(1 + np.exp(z)))) / n + lambda_ * np.sum(
        np.abs(w)
    )
    return loss


def compute_gradient_LR(y, tx, w, lambda_=0):
    """
    Gradient of the logistic regression loss with optional L2 regularization.

    Parameters
    ----------
    y : (np.array) Output data points
    tx : (np.array) Input data points
    w : (np.array) Weights
    lambda_ : (float) Regularization parameter

    Returns
    -------
    (np.array) Gradient vector of shape (d, 1)
    """
    n = tx.shape[0]  # Number of samples
    return (tx.T @ (sigmoid(tx @ w) - y)) / n + 2 * lambda_ * w


def compute_gradient_LR_l1(y, tx, w, lambda_=0):
    """ "
    Gradient of the logistic regression loss with L1 regularization.

    Parameters
    ----------
    y : (np.array) Output data points
    tx : (np.array) Input data points
    w : (np.array) Weights
    lambda_ : (float) Regularization parameter

    Returns
    -------
    (np.array) Gradient vector of shape (d, 1)
    """
    n = tx.shape[0]  # Number of samples
    return (tx.T @ (sigmoid(tx @ w) - y)) / n + lambda_ * np.sign(w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using Gradient Descent (GD).

    Parameters
    ----------
    y : (np.array) Output data points (0 or 1)
    tx : (np.array) Input data points
    initial_w : (np.array) Initial weights
    max_iters : (int) Maximal number of iterations
    gamma : (float) Learning rate

    Returns
    -------
    (np.array, float) Final weights and their corresponding loss
    """
    w = initial_w

    for _ in range(max_iters):
        # Compute gradient and update weights
        gradient = compute_gradient_LR(y, tx, w)
        w = w - gamma * gradient

    # Final loss computation
    final_loss = logistic_loss_function(y, tx, w)

    return w, final_loss


def logistic_regression_with_early_stopping(
    y, tx, initial_w, max_iters, gamma, threshold=1e-8, verbose=True
):
    """
    Logistic regression using Gradient Descent (GD).
    Includes early stopping based on convergence criteria.

    Parameters
    ----------
    y : (np.array) Output data points (0 or 1)
    tx : (np.array) Input data points
    initial_w : (np.array) Initial weights
    max_iters : (int) Maximal number of iterations
    gamma : (float) Learning rate
    threshold : (float) Convergence threshold for early stopping
    verbose : (bool) Whether to print progress messages

    Returns
    -------
    (np.array, float) Final weights and their corresponding loss
    """
    w = initial_w.copy()
    losses = []

    for i in range(max_iters):
        # Compute current loss
        current_loss = logistic_loss_function(y, tx, w)
        losses.append(current_loss)

        # Compute gradient and update weights
        gradient = compute_gradient_LR(y, tx, w)
        w = w - gamma * gradient

        # Check for convergence based on loss change (if we have previous loss)
        if i > 0:
            loss_change = abs(losses[-1] - losses[-2])
            # Stop if loss change is below threshold
            if loss_change < threshold:
                print(f"Converged at iteration {i+1}/{max_iters}")
                print(f"Loss change: {loss_change:.2e}")
                break

        # Optional: Print progress every 100 iterations
        if verbose and (i + 1) % 100 == 0:
            print(f"Iteration {i+1}/{max_iters}, Loss: {current_loss:.6f}")

    # Final loss computation
    final_loss = logistic_loss_function(y, tx, w)

    return w, final_loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using Gradient Descent (GD) with L2 regularization.

    Parameters
    ----------
    y : (np.array) Output data points (0 or 1)
    tx : (np.array) Input data points
    lambda_ : (float) Regularization parameter
    initial_w : (np.array) Initial weights
    max_iters : (int) Maximal number of iterations
    gamma : (float) Learning rate

    Returns
    -------
    (np.array, float) Final weights and their corresponding loss
    """
    w = initial_w

    for _ in range(max_iters):
        # Compute gradient and update weights
        gradient = compute_gradient_LR(y, tx, w, lambda_)
        w = w - gamma * gradient

    # Final loss computation
    final_loss = logistic_loss_function(y, tx, w)

    return w, final_loss


def reg_logistic_regression_with_early_stopping(
    y, tx, lambda_, initial_w, max_iters, gamma, threshold=1e-8, verbose=True
):
    """
    Regularized logistic regression using Gradient Descent (GD) with L2 regularization.
    Includes early stopping based on convergence criteria.

    Parameters
    ----------
    y : (np.array) Output data points (0 or 1)
    tx : (np.array) Input data points
    lambda_ : (float) Regularization parameter
    initial_w : (np.array) Initial weights
    max_iters : (int) Maximal number of iterations
    gamma : (float) Learning rate
    threshold : (float) Convergence threshold for early stopping
    verbose : (bool) Whether to print progress messages

    Returns
    -------
    (np.array, float) Final weights and their corresponding loss
    """
    w = initial_w.copy()
    losses = []

    for i in range(max_iters):
        # Compute current loss
        current_loss = logistic_loss_function(y, tx, w)
        losses.append(current_loss)

        # Compute gradient and update weights
        gradient = compute_gradient_LR(y, tx, w, lambda_)
        w = w - gamma * gradient

        # Check for convergence based on loss change (if we have previous loss)
        if i > 0:
            loss_change = abs(losses[-1] - losses[-2])
            # Stop if loss change is below threshold
            if loss_change < threshold:
                if verbose:
                    print(f"Converged at iteration {i+1}/{max_iters}")
                    print(f"Loss change: {loss_change:.2e}")
                break

        # Optional: Print progress every 100 iterations
        if verbose and (i + 1) % 100 == 0:
            print(f"Iteration {i+1}/{max_iters}, Loss: {current_loss:.6f}")

    # Final loss computation
    final_loss = logistic_loss_function(y, tx, w)

    return w, final_loss


def reg_logistic_regression_l1(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using Gradient Descent (GD) with L1 regularization.
    Includes early stopping based on convergence criteria.

    Parameters
    ----------
    y : (np.array) Output data points (0 or 1)
    tx : (np.array) Input data points
    lambda_ : (float) Regularization parameter
    initial_w : (np.array) Initial weights
    max_iters : (int) Maximal number of iterations
    gamma : (float) Learning rate

    Returns
    -------
    (np.array, float) Final weights and their corresponding loss
    """
    w = initial_w

    for i in range(max_iters):

        # Compute gradient and update weights
        gradient = compute_gradient_LR_l1(y, tx, w, lambda_)
        w = w - gamma * gradient

    # Final loss computation
    final_loss = logisitc_loss_function_l1(y, tx, w)

    return w, final_loss


def reg_logistic_regression_l1_with_early_stopping(
    y, tx, lambda_, initial_w, max_iters, gamma, threshold=1e-8, verbose=True
):
    """
    Regularized logistic regression using Gradient Descent (GD) with L1 regularization.
    Includes early stopping based on convergence criteria.

    Parameters
    ----------
    y : (np.array) Output data points (0 or 1)
    tx : (np.array) Input data points
    lambda_ : (float) Regularization parameter
    initial_w : (np.array) Initial weights
    max_iters : (int) Maximal number of iterations
    gamma : (float) Learning rate
    threshold : (float) Convergence threshold for early stopping
    verbose : (bool) Whether to print progress messages

    Returns
    -------
    (np.array, float) Final weights and their corresponding loss
    """
    w = initial_w.copy()
    losses = []

    for i in range(max_iters):
        # Compute current loss
        current_loss = logisitc_loss_function_l1(y, tx, w)
        losses.append(current_loss)

        # Compute gradient and update weights
        gradient = compute_gradient_LR_l1(y, tx, w, lambda_)
        w = w - gamma * gradient

        # Check for convergence based on loss change (if we have previous loss)
        if i > 0:
            loss_change = abs(losses[-1] - losses[-2])
            # Stop if loss change is below threshold
            if loss_change < threshold:
                if verbose:
                    print(f"Converged at iteration {i+1}/{max_iters}")
                    print(f"Loss change: {loss_change:.2e}")
                break

        # Optional: Print progress every 100 iterations
        if verbose and (i + 1) % 100 == 0:
            print(f"Iteration {i+1}/{max_iters}, Loss: {current_loss:.6f}")

    # Final loss computation
    final_loss = logisitc_loss_function_l1(y, tx, w)

    return w, final_loss


def build_k_indices(y, k_fold, seed):
    """
    Build k indices for k-fold cross validation.

    Parameters
    ----------
    y : (np.array) Output data points
    k_fold : (int) Number of folds
    seed : (int) Random seed

    Returns
    -------
    k_indices : (np.array) k_fold x (N/k_fold) array of indices for each fold

    Examples
    --------
    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def predict_labels(tx, w, cutoff):
    """
    Predict class labels (1 or -1) using model weights and a probability cutoff.

    Parameters
    ----------
    tx : np.array
        Feature matrix of shape (n_samples, n_features).
    w : np.array
        Model weights of shape (n_features,) or (n_features, 1).
    cutoff : float
        Probability threshold for assigning class 1 (otherwise -1).

    Returns
    -------
    np.array
        Predicted labels (1 or -1) of shape (n_samples,).
    """
    probs = sigmoid(tx @ w)
    return np.where(probs >= cutoff, 1, 0)


def precision_recall_f1(y_true, y_pred):
    """
    Compute precision, recall, and F1-score for binary classification.

    Parameters
    ----------
    y_true : np.array
        True labels (1 or -1) of shape (n_samples,).
    y_pred : np.array
        Predicted labels (1 or -1) of shape (n_samples,).

    Returns
    -------
    dict
        Dictionary with keys 'precision', 'recall', and 'f1' (all floats).
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def cross_validation_logistic(
    y,
    tx,
    k_indices,
    normalized_cols_idx,
    degree,
    k,
    lambda_,
    gamma,
    sampling=None,
    factor=1,
    algorithm="l2",
    cutoff=0.5,
    max_iters=1000,
):
    """
    Cross validation for one fold of regularized logistic regression.

    Trains a regularized logistic regression model on the training fold, evaluates on the test fold,
    and computes loss, precision, recall, and F1-score for both sets.

    Parameters
    ----------
    y : (np.array) Labels (0 or 1)
    tx : (np.array) Features with bias column
    k_indices : (np.array) k-fold indices
    normalized_cols_idx : (np.array) Indices of normalized columns for polynomial expansion
    degree : (int) Degree for polynomial expansion
    k : (int) Current fold
    lambda_ : (float) Regularization parameter
    gamma : (float) Learning rate
    sampling : (str or None) Sampling method: None, 'oversampling', or 'undersampling'
    factor : (int) Sampling factor
    algorithm : (str) Algorithm choice: 'l2' for L2 regularization, 'l1' for L1 regularization
    cutoff : (float) Threshold for converting predicted probabilities to class labels (default: 0.5)
    max_iters : (int) Maximum iterations

    Returns
    -------
    (tuple):
        - loss_tr : (float) Training loss
        - loss_te : (float) Test loss
        - metrics_tr : (dict) Precision, recall, and F1-score for training set
        - metrics_te : (dict) Precision, recall, and F1-score for test set
    """
    # Get train and test indices for current fold
    te_indices = k_indices[k]
    tr_indices = np.hstack([k_indices[i] for i in range(len(k_indices)) if i != k])

    # Split data
    y_tr, tx_tr = y[tr_indices], build_poly(tx[tr_indices], normalized_cols_idx, degree)
    y_te, tx_te = y[te_indices], build_poly(tx[te_indices], normalized_cols_idx, degree)

    if sampling == None:
        pass
    elif sampling == "oversampling":
        tx_tr, y_tr = oversampling(tx_tr, y_tr, factor)
    elif sampling == "undersampling":
        tx_tr, y_tr = undersampling(tx_tr, y_tr, factor)
    else:
        raise ValueError("Incorrect sampling method")

    # Initialize weights
    initial_w = np.zeros(tx_tr.shape[1])

    # Train model (silent mode for CV with looser threshold for speed)
    if algorithm == "l2":
        w, _ = reg_logistic_regression(
            y_tr, tx_tr, lambda_, initial_w, max_iters, gamma
        )
    elif algorithm == "l1":
        w, _ = reg_logistic_regression_l1(
            y_tr, tx_tr, lambda_, initial_w, max_iters, gamma
        )
    else:
        raise ValueError("Incorrect algorithm choice")

    y_tr_pred = predict_labels(tx_tr, w, cutoff)
    y_te_pred = predict_labels(tx_te, w, cutoff)

    metrics_tr = precision_recall_f1(y_tr, y_tr_pred)
    metrics_te = precision_recall_f1(y_te, y_te_pred)
    metrics_tr["loss"] = logistic_loss_function(y_tr, tx_tr, w)
    metrics_te["loss"] = logistic_loss_function(y_te, tx_te, w)

    return metrics_tr, metrics_te


def logistic_cross_validation_demo(
    y,
    tx,
    k_fold,
    normalized_cols_idx,
    degrees,
    lambdas,
    gammas,
    samplings,
    factors,
    algorithms,
    cutoffs,
    max_iters=1000,
    seed=42,
    verbose=True,
):
    """
    Perform grid search cross-validation for regularized logistic regression.

    For each combination of degree, lambda, gamma, sampling method, and cutoff, performs k-fold cross-validation
    and computes average training/validation loss and F1-score. Returns the best parameter sets by F1-score and loss.

    Parameters
    ----------
    y : np.array
        True labels (0 or 1) of shape (n_samples,).
    tx : np.array
        Feature matrix (with bias column) of shape (n_samples, n_features).
    k_fold : int
        Number of folds for cross-validation.
    normalized_cols_idx : list or np.array
        Indices of columns to normalize.
    degrees : iterable
        Degrees of polynomial feature expansion to test.
    lambdas : iterable
        Regularization parameters to test.
    gammas : iterable
        Learning rates to test.
    samplings : iterable
        Sampling strategies to test (e.g., None, 'undersample', 'oversample').
    factors : iterable
        Sampling factors to test.
    algorithms : iterable
        Algorithms to test (e.g., 'l2', 'l1').
    cutoffs : iterable
        Cutoff thresholds for classification to test.
    max_iters : int, optional
        Maximum number of iterations for training (default: 1000).
    seed : int, optional
        Random seed for reproducibility (default: 42).
    verbose : bool, optional
        Whether to print progress messages (default: True).

    Returns
    -------
    best_f1_result : dict
        Parameter set and metrics with the highest average validation F1-score.
    best_loss_result : dict
        Parameter set and metrics with the lowest average validation loss.
    results : list of dict
        List of dictionaries with all parameter combinations and their average metrics.
    """

    # Build k-fold indices
    k_indices = build_k_indices(y, k_fold, seed)

    results = []

    for lambda_, gamma, degree, sampling, factor, algorithm, cutoff in product(
        lambdas, gammas, degrees, samplings, factors, algorithms, cutoffs
    ):

        if (
            sampling == None
            and factor != 1
            or sampling == "undersampling"
            and factor < 1
        ):
            break

        if verbose:
            print("-----------------------------------------")
            print("Hyperparameter combination:\n")
            print(
                f"Testing lambda={lambda_:.2e}\n gamma={gamma:.2e}\n degree={degree}\n sampling={sampling}\n cutoff={cutoff}\n algorithm={algorithm}\n factor={factor}\n"
            )

        params = {
            "lambda": lambda_,
            "gamma": gamma,
            "degree": degree,
            "sampling": sampling,
            "factor": factor,
            "algorithm": algorithm,
            "cutoff": cutoff,
        }

        total_loss_tr = 0
        total_loss_te = 0
        total_f1_tr = 0
        total_f1_te = 0

        for k in range(k_fold):

            metrics_tr, metrics_te = cross_validation_logistic(
                y,
                tx,
                k_indices,
                normalized_cols_idx,
                degree,
                k,
                lambda_,
                gamma,
                sampling,
                factor,
                algorithm,
                cutoff,
                max_iters,
            )
            total_loss_tr += metrics_tr["loss"]
            total_loss_te += metrics_te["loss"]
            total_f1_tr += metrics_tr["f1"]
            total_f1_te += metrics_te["f1"]

        results.append(
            {
                **params,
                "avg_loss_tr": total_loss_tr / k_fold,
                "avg_loss_te": total_loss_te / k_fold,
                "avg_f1_tr": total_f1_tr / k_fold,
                "avg_f1_te": total_f1_te / k_fold,
            }
        )

        if verbose:
            print(
                f"Avg Train Loss: {total_loss_tr / k_fold:.4f}\n Avg Test Loss: {total_loss_te / k_fold:.4f}\n Avg Train F1: {total_f1_tr / k_fold:.4f}\n Avg Test F1: {total_f1_te / k_fold:.4f}\n"
            )
            print("-----------------------------------------------------")

    avg_f1_te_array = np.array([res["avg_f1_te"] for res in results])
    avg_loss_te_array = np.array([res["avg_loss_te"] for res in results])

    # Best F1 (maximize)
    best_f1_idx = np.argmax(avg_f1_te_array)
    best_f1_result = results[best_f1_idx]

    # Best loss (minimize)
    best_loss_idx = np.argmin(avg_loss_te_array)
    best_loss_result = results[best_loss_idx]

    print("Best by F1 (avg_f1_te):")
    print(best_f1_result)

    print("\nBest by Loss (avg_loss_te):")
    print(best_loss_result)

    return best_f1_result, best_loss_result, results
