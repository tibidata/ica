import numpy as np
import matplotlib.pyplot as plt
from src import functions as func


class IndependentComponentAnalysis:
    def __init__(self, X: np.array):
        """
        :param X: ndarray: Data matrix of the mixed signals
        """
        self.S = None  # De-mixed signals, used later for plotting
        self.X = X
        self.cov = None

    def predict(self, max_iterations: int = 5000, tolerance: float = 1e-5):
        """
        Implementation of FastICA using negentropy as a determining function of gaussanity
        :param max_iterations: float : number of maximum iterations for the algorithm
        :param tolerance: float : used to determine if the algorithm converged
        :return: ndarray : De-mixed array of the original signals
        """
        X_centered = func.center(X=self.X)  # Centering the data matrix
        X_whitened = func.whiten(X=X_centered)  # Whitening the data matrix

        ics = X_whitened.shape[0]  # Number of independent components we are looking for

        W = np.zeros((ics, ics))  # Creating a dummy de-mixing matrix for the signals

        for p in range(ics):
            wp = np.random.rand(ics)  # Creating a random pth row for the de-mixing matrix

            for j in range(max_iterations):
                w_new = func.calculate_new_wp(X=X_whitened, wp=wp)  # Calculating new pth row of the de-mixing matrix

                if p >= 1:
                    w_new -= np.dot(np.dot(w_new, W[:p].T), W[:p])

                dist = np.abs(np.abs((wp * w_new).sum()) - 1)

                wp = w_new

                if dist < tolerance:  # Checking if the distance below tolerance ( converged or not)
                    break
            W[p, :] = wp

        S = np.dot(W, X_whitened)  # De-mixing data matrix

        self.S = S

        return S

    def plot(self, O: np.array = None):
        """
        Function the plot the mixed and the predicted de-mixed signals
        :param O: Original signals matrix if it is available
        :return: None
        """
        if self.S is None:  # Raise exception if the model wasn't ran first
            raise Exception('Please use the predict() function first')
        else:
            fig = plt.figure()

            plt.subplot(3, 1, 1)
            for x in self.X:
                plt.plot(x)
                plt.title('Mixed Signals')

            plt.subplot(3, 1, 2)
            for s in self.S:
                plt.plot(s)
                plt.title('Predicted de-mixed signals')

            if O is not None:  # If the original signal matrix is available plots that as well for comparison

                plt.subplot(3, 1, 3)
                for o in O:
                    plt.plot(o)
                    plt.title('Original signals')

            fig.tight_layout()
            plt.show()
