"""Built-in metrics
"""
import torch


def _check_lenght(y_true, y_pred):
    assert len(y_true) == len(y_pred), "Prediction and target should have equal length."


def binary_accuracy(y_true, y_pred):
    _check_lenght(y_true, y_pred)
    return torch.mean((y_pred == y_true).float())


def categorical_accuracy(y_true, y_pred):
    _check_lenght(y_true, y_pred)
    _, preds = torch.max(y_pred, 1)
    return torch.mean((preds == y_true).float())
