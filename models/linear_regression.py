import numpy as np


class LinearRegression:

    def __init__(self, lr=0.01, n_iter=1000, lr_schedule="constant", decay=0.001):
        self.lr = lr
        self.n_iter = n_iter
        self.lr_schedule = lr_schedule
        self.decay = decay
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _get_lr(self, t):
        # adjust the learning rate depending on the chosen schedule
        if self.lr_schedule == "time_decay":
            return self.lr / (1 + self.decay * t)
        elif self.lr_schedule == "step_decay":
            # drop the lr every 100 steps
            factor = np.floor((1 + t) / 100)
            return self.lr * (self.decay ** factor)
        else:
            return self.lr  # constant, no change

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        n_samples, n_features = X.shape

        # start weights at zero
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for i in range(self.n_iter):
            current_lr = self._get_lr(i)

            # predictions with current weights
            y_hat = X @ self.weights + self.bias

            # difference between predicted and actual
            error = y_hat - y

            # track MSE so we can see if the model is converging
            self.loss_history.append(np.mean(error ** 2))

            # compute gradients
            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)

            # gradient descent update step
            self.weights -= current_lr * dw
            self.bias -= current_lr * db

        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        return X @ self.weights + self.bias

    def score(self, X, y):
        # R² score — how much of the variance our model explains
        y_hat = self.predict(X)
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            return 0.0
        return 1 - (ss_res / ss_tot)

    def get_params(self):
        return {"weights": self.weights, "bias": self.bias}