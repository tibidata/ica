import numpy as np


def center(X: np.array):
    """
    Function to center matrix for further use
    :return: A centered X matrix as a numpy array
    """
    return X - np.mean(X, axis=1, keepdims=True)


def whiten(X: np.array):
    """
    Function to whiten the data matrix using SVD which is a part of data preprocessing
    :return: A whitened numpy array of X
    """

    X_centered = center(X=X)
    x_covariance = np.cov(X_centered)

    U, S, V = np.linalg.svd(x_covariance)
    X_whitened = np.dot(np.dot(U, np.diag(1.0 / np.sqrt(S))), U.T).dot(X_centered)
    return X_whitened


def g(X: np.array, alpha: float = 1.0):
    """
    Function to calculate negentropy
    :param X: ndarray : Data matrix of the signals
    :param alpha: float : Regularization parameter by default 1.0
    """
    return np.tanh(alpha * X)


def g_der(X: np.array, alpha: float = 1.0):
    """
    Derivative of the g function
    :param X: ndarray : Data matrix of the signals
    :param alpha: Regularization parameter by default 1.0
    """
    g_x = g(X=X)
    return alpha * (1 - g_x ** 2)


def calculate_new_wp(X: np.array, wp: np.array):
    """
    Used for approximating the pth row of the de-mixing matrix
    :param X: ndarray : Data matrix of the signals
    :param wp: ndarray: Previous pth row of the de-mixing matrix
    :return: ndarray: New pth row of the de-mixing matrix
    """
    wp_new = (X * g(np.dot(wp.T, X))).mean(axis=1) - g_der(np.dot(wp.T, X)).mean() * wp
    wp_new /= np.sqrt((wp_new ** 2).sum())
    return wp_new
