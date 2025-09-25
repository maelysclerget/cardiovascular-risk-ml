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
    n = tx.shape[0] # Number of samples
    e = y - tx@w
    return (e.T @ e)/(2*n)

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
    n = tx.shape[0] # Number of samples
    e = y - tx@w
    return -(tx.T @ e)/n 

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
    loss = np.inf
    w = initial_w

    for _ in range(max_iters):
        gradient, loss = compute_gradient_MSE(y, tx, w), mean_squared_error_loss(y, tx, w)
        w = w - gamma*gradient

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
    loss = np.inf
    w = initial_w
    n = tx.shape[0] # Number of samples

    for _ in range(max_iters):
        ind = np.random.randint(low = 0, high = n) # Randomly selected index/data point for gradient and loss computation
        gradient = compute_gradient_MSE(y[ind, np.newaxis, :], tx[ind, np.newaxis, :], w)
        loss = mean_squared_error_loss(y[ind, np.newaxis, :], tx[ind, np.newaxis, :], w)
        w = w - gamma*gradient

    return w, loss