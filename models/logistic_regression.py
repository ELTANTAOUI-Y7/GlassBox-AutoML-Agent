import numpy as np


class LogisticRegression:

    def __init__(self, lr=0.01, n_iter=1000, lr_schedule="constant", decay=0.001, threshold=0.5):
        self.lr = lr
        self.n_iter = n_iter
        self.lr_schedule = lr_schedule
        self.decay = decay
        self.threshold = threshold
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _get_lr(self, t):
        if self.lr_schedule == "time_decay":
            return self.lr / (1 + self.decay * t)
        elif self.lr_schedule == "step_decay":
            factor = np.floor((1 + t) / 100)
            return self.lr * (self.decay ** factor)
        else:
            return self.lr

    def _sigmoid(self, z):
        # clip z to avoid overflow in exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for i in range(self.n_iter):
            current_lr = self._get_lr(i)

            # forward pass — get probabilities
            z = X @ self.weights + self.bias
            y_hat = self._sigmoid(z)

            # binary cross-entropy loss
            eps = 1e-15  # avoid log(0)
            loss = -np.mean(
                y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)
            )
            self.loss_history.append(loss)

            # gradients (same form as linear regression — pretty neat)
            error = y_hat - y
            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error)

            # update weights
            self.weights -= current_lr * dw
            self.bias -= current_lr * db

        return self

    def predict_proba(self, X):
        # returns the raw probability for the positive class
        X = np.array(X, dtype=float)
        z = X @ self.weights + self.bias
        return self._sigmoid(z)

    def predict(self, X):
        # convert probabilities to 0/1 labels using the threshold
        proba = self.predict_proba(X)
        return (proba >= self.threshold).astype(int)

    def score(self, X, y):
        # simple accuracy
        y_pred = self.predict(X)
        return np.mean(y_pred == np.array(y))

    def get_params(self):
        return {"weights": self.weights, "bias": self.bias}
