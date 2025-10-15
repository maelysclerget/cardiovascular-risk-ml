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
        gradient = compute_gradient_LR(y, tx, w)
        w = w - gamma * gradient

    loss = logistic_loss_function(y, tx, w)

    return w, loss


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
        gradient = compute_gradient_LR(y, tx, w, lambda_)
        w = w - gamma * gradient

    loss = logistic_loss_function(y, tx, w)

    return w, loss
