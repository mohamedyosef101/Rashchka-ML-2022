import numpy as np

class Perceptron:
    """
    Perceptron classifier.
    
    Methods:
        fit(X, y): Fit the model to the training data.
        net_input(X): Compute the net input. WX + b.
        predict(X): Return class label after applying unit step function.
    Args:
        eta (float): Learning rate.
        epochs (int): Number of passes over the training dataset.
        random_state (int): Random number generator seed for reproducibility.
        store_results (bool): stores the weights and bias into a list.
    Model parameters:
        w_ (ndarray): Weights after fitting.
        b_ (float): Bias after fitting.
        converged (bool): True if the algorithm converged.
    Stored results:
        weights (list): Weights after each epoch.
        biases (list): Biases after each epoch.
        errors (list): Number of incorrect classifications after each epoch.
    """
    def __init__(self, eta=0.01, epochs=50, random_state=101, store_results=False):
        self.eta = eta
        self.epochs = epochs
        self.random_state = np.random.RandomState(random_state)
        self.store_results = store_results
        self.converged = False

    def fit(self, X, y):
        # Initialize weights and bias
        self.w_ = self.random_state.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = 0.0

        if self.store_results:
            self.weights = []
            self.biases = [] 
            self.errors = []
        
        for epoch in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                if prediction != target:
                    # Update weights and bias according to Perceptron learning rule
                    update = self.eta * (target - prediction)
                    self.w_ += update * xi
                    self.b_ += update
                    errors += 1
            if self.store_results:
                self.weights.append(self.w_.copy())
                self.biases.append(self.b_)
                self.errors.append(errors)

            # Convergence check: if no errors, the algorithm has converged
            if epoch > 0 and errors == 0:
                self.converged = True
                break
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        # Threshold function: returns 1 if net_input >= 0, otherwise 0
        return np.where(self.net_input(X) >= 0.0, 1, 0)
