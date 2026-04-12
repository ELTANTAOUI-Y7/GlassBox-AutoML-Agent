import numpy as np


class KNN:

    def __init__(self, k=5, task="classification", metric="euclidean"):
        self.k = k
        self.task = task       # "classification" or "regression"
        self.metric = metric   # "euclidean" or "manhattan"
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # KNN doesn't actually train — it just memorizes the data
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y)
        return self

    def _compute_distances(self, x):
        if self.metric == "manhattan":
            # sum of absolute differences
            return np.sum(np.abs(self.X_train - x), axis=1)
        else:
            # euclidean: sqrt of sum of squared differences
            return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

    def _predict_single(self, x):
        distances = self._compute_distances(x)

        # get the k nearest neighbor indices
        k_indices = np.argsort(distances)[: self.k]
        k_labels = self.y_train[k_indices]

        if self.task == "regression":
            # average the values of the neighbors
            return np.mean(k_labels)
        else:
            # majority vote — pick the most common class
            classes, counts = np.unique(k_labels, return_counts=True)
            return classes[np.argmax(counts)]

    def predict(self, X):
        X = np.array(X, dtype=float)
        return np.array([self._predict_single(x) for x in X])

    def score(self, X, y):
        y_pred = self.predict(X)
        y = np.array(y)

        if self.task == "regression":
            # R² score
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            if ss_tot == 0:
                return 0.0
            return 1 - (ss_res / ss_tot)
        else:
            return np.mean(y_pred == y)
