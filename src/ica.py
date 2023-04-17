import numpy as np
from scipy import linalg


class IndependentComponentAnalysis:
    def __init__(self, X: np.array):
        self.X = X

    def center(self):
        """
        Function to center matrix for further use
        :return: A centered X matrix as a numpy array
        """
        return self.X.T - np.mean(self.X.T, axis=0)

    def whiten(self):
        """
        Function to whiten the data matrix which is a part of data preprocessing
        :return: A whitened numpy array of X
        """
        x_centered = self.center()  # Centering matrix
        x_covariance = np.cov(x_centered, rowvar=True, bias=True)  # Calculating covariance matrix
        eig_val, eig_vect = linalg.eigh(x_covariance)  # Calculating eigen values and vectors

        diagonal_eig_val = np.diag(1 / ((eig_val + .1e-5) ** 0.5))
        diagonal_eig_val = diagonal_eig_val.real.round(4)

        whitening_matrix = np.dot(np.dot(eig_vect, diagonal_eig_val), eig_vect.T)  # Calculating whitening matrix

        x_whitened = self.X @ whitening_matrix  # Calculating whited matrix

        return x_whitened
