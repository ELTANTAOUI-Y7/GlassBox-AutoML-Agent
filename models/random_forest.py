import numpy as np
from models.decision_tree import DecisionTree


class RandomForest:

    def __init__(self, n_trees=100, task="classification", criterion=None,
                 max_depth=5, min_samples_split=2, random_state=None):
        self.n_trees = n_trees
        self.task = task
        # default criterion based on task if not specified
        self.criterion = criterion if criterion else ("gini" if task == "classification" else "mse")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape
        # forest uses sqrt(n_features) at each split — key to Random Forest
        n_feature_subset = int(np.sqrt(n_features))

        self.trees = []

        for _ in range(self.n_trees):
            # bootstrap sample — sample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            # random feature subset for this tree
            feature_indices = np.random.choice(n_features, size=n_feature_subset, replace=False)
            X_subset = X_boot[:, feature_indices]

            tree = DecisionTree(
                task=self.task,
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X_subset, y_boot)

            # store both the tree and which features it was trained on
            self.trees.append((tree, feature_indices))

        return self

    def predict(self, X):
        X = np.array(X, dtype=float)

        # collect predictions from each tree
        all_preds = np.array([
            tree.predict(X[:, feature_indices])
            for tree, feature_indices in self.trees
        ])  # shape: (n_trees, n_samples)

        if self.task == "regression":
            # average across all trees
            return np.mean(all_preds, axis=0)
        else:
            # majority vote for each sample
            predictions = []
            for sample_preds in all_preds.T:
                classes, counts = np.unique(sample_preds, return_counts=True)
                predictions.append(classes[np.argmax(counts)])
            return np.array(predictions)

    def score(self, X, y):
        y_pred = self.predict(X)
        y = np.array(y)
        if self.task == "regression":
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            if ss_tot == 0:
                return 0.0
            return 1 - (ss_res / ss_tot)
        return np.mean(y_pred == y)
