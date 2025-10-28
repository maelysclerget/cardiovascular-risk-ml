import numpy as np


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
    (np.array) Gradient vector of shape (d, 1), given by:
    """
    n = tx.shape[0]  # Number of samples
    return (tx.T @ (sigmoid(tx @ w) - y)) / n + 2 * lambda_ * w


def logistic_regression(y, tx, initial_w, max_iters, gamma, threshold=1e-8):
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
        if (i + 1) % 100 == 0:
            print(f"Iteration {i+1}/{max_iters}, Loss: {current_loss:.6f}")
    
    # Final loss computation
    final_loss = logistic_loss_function(y, tx, w)
    
    return w, final_loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, threshold=1e-8, verbose=True):
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

    Returns
    -------
    (np.array, float) Final weights and their corresponding loss
    """
    w = initial_w.copy()
    losses = []
    
    for i in range(max_iters):
        # Compute current loss
        current_loss = logistic_loss_function(y, tx, w, lambda_)
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
    final_loss = logistic_loss_function(y, tx, w, lambda_)
    
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
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation_logistic(y, tx, k_indices, k, lambda_, gamma, max_iters=1000, initial_w=None):
    """
    Cross validation for one fold of regularized logistic regression.
    
    Parameters
    ----------
    y : (np.array) Labels (0 or 1)
    tx : (np.array) Features with bias column
    k_indices : (np.array) k-fold indices
    k : (int) Current fold
    lambda_ : (float) Regularization parameter
    gamma : (float) Learning rate
    max_iters : (int) Maximum iterations
    initial_w : (np.array) Initial weights. If None, will use zeros
    
    Returns
    -------
    loss_tr : (float) Training loss
    loss_te : (float) Test loss
    """
    # Get train and test indices for current fold
    te_indices = k_indices[k]
    tr_indices = np.hstack([k_indices[i] for i in range(len(k_indices)) if i != k])
    
    # Split data
    y_tr, tx_tr = y[tr_indices], tx[tr_indices]
    y_te, tx_te = y[te_indices], tx[te_indices]
    
    # Initialize weights
    if initial_w is None:
        initial_w = np.zeros(tx_tr.shape[1])
    
    # Train model (silent mode for CV with looser threshold for speed)
    w, _ = reg_logistic_regression(y_tr, tx_tr, lambda_, initial_w, max_iters, gamma, threshold=1e-4, verbose=False)
    
    # Compute losses
    loss_tr = logistic_loss_function(y_tr, tx_tr, w, lambda_)
    loss_te = logistic_loss_function(y_te, tx_te, w, lambda_)
    
    return loss_tr, loss_te


def logistic_cross_validation_demo(y, tx, k_fold, lambdas, gammas, max_iters=1000, seed=12, initial_w=None):
    """
    Cross validation for regularized logistic regression over lambda and gamma parameters.
    
    Parameters
    ----------
    y : (np.array) Labels (0 or 1) 
    tx : (np.array) Features with bias column
    k_fold : (int) Number of folds
    lambdas : (np.array) Array of regularization parameters to test
    gammas : (np.array) Array of learning rates to test
    max_iters : (int) Maximum iterations for training
    seed : (int) Random seed
    initial_w : (np.array) Initial weights to use. If None, will use zeros
    
    Returns
    -------
    best_lambda : (float) Best regularization parameter
    best_gamma : (float) Best learning rate  
    best_loss : (float) Best validation loss
    results : (dict) Dictionary containing all results
    """
    
    # Build k-fold indices
    k_indices = build_k_indices(y, k_fold, seed)
    
    # Initialize results storage
    results = {
        'lambdas': lambdas,
        'gammas': gammas,
        'train_losses': np.zeros((len(lambdas), len(gammas))),
        'test_losses': np.zeros((len(lambdas), len(gammas)))
    }
    
    best_loss = float('inf')
    best_lambda = None
    best_gamma = None
    
    print(f"Starting cross-validation with {len(lambdas)} lambdas and {len(gammas)} gammas...")
    
    # Grid search over lambda and gamma
    for i, lambda_ in enumerate(lambdas):
        for j, gamma in enumerate(gammas):
            print(f"Testing lambda={lambda_:.1e}, gamma={gamma:.1e}")
            
            # Accumulate losses across folds
            loss_tr_total = 0
            loss_te_total = 0
            
            for fold in range(k_fold):
                loss_tr, loss_te = cross_validation_logistic(
                    y, tx, k_indices, fold, lambda_, gamma, max_iters, initial_w=initial_w
                )
                loss_tr_total += loss_tr
                loss_te_total += loss_te
            
            # Average losses across folds
            avg_loss_tr = loss_tr_total / k_fold
            avg_loss_te = loss_te_total / k_fold
            
            # Store results
            results['train_losses'][i, j] = avg_loss_tr
            results['test_losses'][i, j] = avg_loss_te
            
            # Update best parameters
            if avg_loss_te < best_loss:
                best_loss = avg_loss_te
                best_lambda = lambda_
                best_gamma = gamma
                
            print(f"  Avg train loss: {avg_loss_tr:.4f}, Avg test loss: {avg_loss_te:.4f}")
    
    print(f"\nBest parameters:")
    print(f"Lambda: {best_lambda:.1e}")
    print(f"Gamma: {best_gamma:.1e}")  
    print(f"Best validation loss: {best_loss:.4f}")
    
    return best_lambda, best_gamma, best_loss, results

