"""Evaluation functions for TextING model"""

import numpy as np
from sklearn import metrics


def eval(y_pred, y_true):
    """
    Evaluate predictions against true labels.

    Args:
        y_pred: Predicted labels (numpy array)
        y_true: True labels (numpy array)

    Returns:
        Dictionary of evaluation metrics
    """
    accuracy = metrics.accuracy_score(y_true, y_pred)
    f1_weighted = metrics.f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_macro = metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = metrics.f1_score(y_true, y_pred, average='micro', zero_division=0)
    precision_weighted = metrics.precision_score(y_true, y_pred, average='weighted', zero_division=0)
    precision_macro = metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
    precision_micro = metrics.precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall_weighted = metrics.recall_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_macro = metrics.recall_score(y_true, y_pred, average='macro', zero_division=0)
    recall_micro = metrics.recall_score(y_true, y_pred, average='micro', zero_division=0)

    return {
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "precision_weighted": precision_weighted,
        "precision_macro": precision_macro,
        "precision_micro": precision_micro,
        "recall_weighted": recall_weighted,
        "recall_macro": recall_macro,
        "recall_micro": recall_micro
    }
