import numpy as np


class GaussianNaiveBayes:

    def __init__(self, laplace_smoothing=1e-9):
        self.laplace_smoothing = laplace_smoothing
        self.classes = None
        self.priors = {}       # P(class)
        self.means = {}        # mean per feature per class
        self.variances = {}    # variance per feature per class

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)

        self.classes = np.unique(y)
        n_samples = X.shape[0]

        for c in self.classes:
            # grab all rows belonging to this class
            X_c = X[y == c]

            # prior probability for this class
            self.priors[c] = X_c.shape[0] / n_samples

            # per-feature statistics for this class
            self.means[c] = np.mean(X_c, axis=0)
            # add Laplace smoothing to variance so we never divide by zero
            self.variances[c] = np.var(X_c, axis=0) + self.laplace_smoothing

        return self

    def _class_log_likelihood(self, x, c):
        mean = self.means[c]
        var = self.variances[c]

        # Gaussian probability density in log form (more numerically stable)
        log_prob = -0.5 * np.sum(np.log(2 * np.pi * var))
        log_prob -= 0.5 * np.sum(((x - mean) ** 2) / var)

        return np.log(self.priors[c]) + log_prob

    def predict_proba(self, X):
        X = np.array(X, dtype=float)
        # compute raw log-likelihoods for each class, then softmax to get probs
        log_likelihoods = np.array(
            [[self._class_log_likelihood(x, c) for c in self.classes] for x in X]
        )
        # numerically stable softmax
        log_likelihoods -= np.max(log_likelihoods, axis=1, keepdims=True)
        proba = np.exp(log_likelihoods)
        proba /= np.sum(proba, axis=1, keepdims=True)
        return proba

    def predict(self, X):
        X = np.array(X, dtype=float)
        # for each sample, pick the class with the highest log-likelihood
        predictions = [
            self.classes[np.argmax([self._class_log_likelihood(x, c) for c in self.classes])]
            for x in X
        ]
        return np.array(predictions)

    def score(self, X, y):
        return np.mean(self.predict(X) == np.array(y))
