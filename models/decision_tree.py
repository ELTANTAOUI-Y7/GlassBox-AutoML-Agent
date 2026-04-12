import numpy as np


class Node:
    # a single node in the tree — either a split or a leaf
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature       # which feature to split on
        self.threshold = threshold   # the split value
        self.left = left             # left subtree (feature <= threshold)
        self.right = right           # right subtree (feature > threshold)
        self.value = value           # only set for leaf nodes


class DecisionTree:

    def __init__(self, task="classification", criterion="gini", max_depth=5, min_samples_split=2):
        self.task = task
        self.criterion = criterion           # "gini" for classif., "mse" for regression
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    # ------------------------------------------------------------------
    # Impurity measures
    # ------------------------------------------------------------------

    def _gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def _mse(self, y):
        if len(y) == 0:
            return 0
        return np.var(y)  # variance == MSE around the mean

    def _impurity(self, y):
        if self.criterion == "mse":
            return self._mse(y)
        return self._gini(y)

    # ------------------------------------------------------------------
    # Splitting logic
    # ------------------------------------------------------------------

    def _best_split(self, X, y):
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        parent_impurity = self._impurity(y)
        n_samples = len(y)

        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                # weighted impurity after the split
                weighted_impurity = (
                    len(y_left) / n_samples * self._impurity(y_left)
                    + len(y_right) / n_samples * self._impurity(y_right)
                )

                gain = parent_impurity - weighted_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    # ------------------------------------------------------------------
    # Recursive tree builder
    # ------------------------------------------------------------------

    def _build_tree(self, X, y, depth=0):
        # stopping conditions
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return self._make_leaf(y)

        feature, threshold = self._best_split(X, y)

        # if no useful split exists, make a leaf
        if feature is None:
            return self._make_leaf(y)

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature=feature, threshold=threshold, left=left_subtree, right=right_subtree)

    def _make_leaf(self, y):
        if self.task == "regression":
            value = np.mean(y)
        else:
            # most common class
            classes, counts = np.unique(y, return_counts=True)
            value = classes[np.argmax(counts)]
        return Node(value=value)

    # ------------------------------------------------------------------
    # Fit / Predict
    # ------------------------------------------------------------------

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)
        self.root = self._build_tree(X, y)
        return self

    def _traverse(self, x, node):
        # leaf node — return its value
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict(self, X):
        X = np.array(X, dtype=float)
        return np.array([self._traverse(x, self.root) for x in X])

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
