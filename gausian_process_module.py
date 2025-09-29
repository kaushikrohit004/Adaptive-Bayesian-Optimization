"""
Gaussian Process Module
"""

import numpy as np

class GaussianProcess:
    def __init__(self):
        """
        Initialize GP with empty training data.
        """
        self.X_train = []
        self.Y_train = []

    def predict(self, x):
        """
        Predict the mean and variance at a given point x.
        returns:
        - mean: average of observed values
        - variance: standard deviation of observed values (with small noise)
        """
        if self.Y_train:
            mu = np.mean(self.Y_train)
            sigma = np.std(self.Y_train) + 1e-6  
        else:
            mu = np.random.random()  
            sigma = np.random.random() * 0.1  
        return mu, sigma

    def update(self, X, Y):
        """
        Update the GP with new training data.
        """
        self.X_train = X
        self.Y_train = Y
