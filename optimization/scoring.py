"""
glassbox.optimization.scoring
==============================

**Evaluation Metrics** — scoring functions for use with the optimization
search strategies and standalone model evaluation.

Every metric is implemented from scratch using NumPy arithmetic — no
Scikit-Learn or SciPy.  All functions follow the same signature::

    score = metric(y_true, y_pred) -> float

Search strategies (GridSearch, RandomSearch) treat higher scores as better.
For error metrics (MSE, MAE) use the negated variants
(:func:`neg_mean_squared_error`, :func:`neg_mean_absolute_error`) so the
search can maximise them consistently.

Classification Metrics
-----------------------
accuracy_score          – Fraction of correctly classified samples.
precision_score         – TP / (TP + FP), binary.
recall_score            – TP / (TP + FN), binary.
f1_score                – Harmonic mean of precision and recall, binary.
confusion_matrix        – Full K×K count matrix.

Regression Metrics
------------------
mean_squared_error      – Average squared residual.
mean_absolute_error     – Average absolute residual.
r2_score                – Coefficient of determination R².

Negated Variants (for maximisation)
------------------------------------
neg_mean_squared_error  – ``-MSE``
neg_mean_absolute_error – ``-MAE``
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_arrays(
    y_true: Any,
    y_pred: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert inputs to NumPy arrays and validate shape compatibility."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_true.shape} "
            f"but y_pred has shape {y_pred.shape}."
        )
    if y_true.ndim != 1:
        raise ValueError(
            f"Expected 1-D arrays, got ndim={y_true.ndim}."
        )
    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Classification Metrics
# ---------------------------------------------------------------------------

def accuracy_score(y_true: Any, y_pred: Any) -> float:
    """Fraction of correctly classified samples.

    .. math::

        \\text{accuracy} = \\frac{1}{n} \\sum_{i=1}^{n}
        \\mathbf{1}[y_i = \\hat{y}_i]

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth class labels.
    y_pred : array-like of shape (n_samples,)
        Predicted class labels.

    Returns
    -------
    float
        Accuracy in ``[0, 1]``.
    """
    y_true, y_pred = _to_arrays(y_true, y_pred)
    return float(np.sum(y_true == y_pred) / y_true.size)


def precision_score(
    y_true: Any,
    y_pred: Any,
    pos_label: Any = 1,
) -> float:
    """Binary precision — fraction of positive predictions that are correct.

    .. math::

        \\text{precision} = \\frac{TP}{TP + FP}

    Returns 0.0 when there are no positive predictions.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    pos_label : scalar, optional
        The label of the positive class.  Default 1.

    Returns
    -------
    float
        Precision in ``[0, 1]``.
    """
    y_true, y_pred = _to_arrays(y_true, y_pred)
    tp = int(np.sum((y_pred == pos_label) & (y_true == pos_label)))
    fp = int(np.sum((y_pred == pos_label) & (y_true != pos_label)))
    denom = tp + fp
    return float(tp / denom) if denom > 0 else 0.0


def recall_score(
    y_true: Any,
    y_pred: Any,
    pos_label: Any = 1,
) -> float:
    """Binary recall — fraction of actual positives that are retrieved.

    .. math::

        \\text{recall} = \\frac{TP}{TP + FN}

    Returns 0.0 when there are no positive ground-truth samples.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    pos_label : scalar, optional
        The label of the positive class.  Default 1.

    Returns
    -------
    float
        Recall in ``[0, 1]``.
    """
    y_true, y_pred = _to_arrays(y_true, y_pred)
    tp = int(np.sum((y_pred == pos_label) & (y_true == pos_label)))
    fn = int(np.sum((y_pred != pos_label) & (y_true == pos_label)))
    denom = tp + fn
    return float(tp / denom) if denom > 0 else 0.0


def f1_score(
    y_true: Any,
    y_pred: Any,
    pos_label: Any = 1,
) -> float:
    """Binary F1-Score — harmonic mean of precision and recall.

    .. math::

        F_1 = \\frac{2 \\cdot \\text{precision} \\cdot \\text{recall}}
                    {\\text{precision} + \\text{recall}}

    Returns 0.0 when both precision and recall are zero.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    pos_label : scalar, optional
        The label of the positive class.  Default 1.

    Returns
    -------
    float
        F1-Score in ``[0, 1]``.
    """
    p = precision_score(y_true, y_pred, pos_label)
    r = recall_score(y_true, y_pred, pos_label)
    denom = p + r
    return float(2.0 * p * r / denom) if denom > 0.0 else 0.0


def confusion_matrix(y_true: Any, y_pred: Any) -> np.ndarray:
    """Compute the K×K confusion matrix from scratch.

    Rows represent true classes; columns represent predicted classes.
    ``cm[i, j]`` is the number of samples of true class *i* predicted as
    class *j*.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.

    Returns
    -------
    np.ndarray of shape (n_classes, n_classes) and dtype int
        Confusion matrix.
    """
    y_true, y_pred = _to_arrays(y_true, y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    label_to_idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm


# ---------------------------------------------------------------------------
# Regression Metrics
# ---------------------------------------------------------------------------

def mean_squared_error(y_true: Any, y_pred: Any) -> float:
    """Mean Squared Error (MSE).

    .. math::

        \\text{MSE} = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2

    Parameters
    ----------
    y_true : array-like
        Ground-truth continuous targets.
    y_pred : array-like
        Model predictions.

    Returns
    -------
    float
        Non-negative MSE.
    """
    y_true, y_pred = _to_arrays(y_true, y_pred)
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    residuals = y_true - y_pred
    return float(np.sum(residuals ** 2) / y_true.size)


def mean_absolute_error(y_true: Any, y_pred: Any) -> float:
    """Mean Absolute Error (MAE).

    .. math::

        \\text{MAE} = \\frac{1}{n} \\sum_{i=1}^{n} |y_i - \\hat{y}_i|

    Parameters
    ----------
    y_true : array-like
        Ground-truth continuous targets.
    y_pred : array-like
        Model predictions.

    Returns
    -------
    float
        Non-negative MAE.
    """
    y_true, y_pred = _to_arrays(y_true, y_pred)
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    return float(np.sum(np.abs(y_true - y_pred)) / y_true.size)


def r2_score(y_true: Any, y_pred: Any) -> float:
    """Coefficient of Determination (R²).

    .. math::

        R^2 = 1 - \\frac{SS_{\\text{res}}}{SS_{\\text{tot}}}
            = 1 - \\frac{\\sum (y_i - \\hat{y}_i)^2}
                        {\\sum (y_i - \\bar{y})^2}

    Returns 1.0 for a perfect predictor.  Returns 0.0 when the model
    performs no better than always predicting the mean.  Can be negative
    for a model worse than a constant mean predictor.

    When ``SS_tot = 0`` (constant target), returns 1.0 if ``SS_res = 0``
    (exact fit), otherwise 0.0.

    Parameters
    ----------
    y_true : array-like
        Ground-truth continuous targets.
    y_pred : array-like
        Model predictions.

    Returns
    -------
    float
    """
    y_true, y_pred = _to_arrays(y_true, y_pred)
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    y_mean = float(np.sum(y_true) / y_true.size)
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Negated variants for maximisation-based search
# ---------------------------------------------------------------------------

def neg_mean_squared_error(y_true: Any, y_pred: Any) -> float:
    """Negated MSE — use this when the search strategy **maximises** its score.

    Returns
    -------
    float
        ``-mean_squared_error(y_true, y_pred)``
    """
    return -mean_squared_error(y_true, y_pred)


def neg_mean_absolute_error(y_true: Any, y_pred: Any) -> float:
    """Negated MAE — use this when the search strategy **maximises** its score.

    Returns
    -------
    float
        ``-mean_absolute_error(y_true, y_pred)``
    """
    return -mean_absolute_error(y_true, y_pred)
