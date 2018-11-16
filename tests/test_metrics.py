import sys
import torch
from smartai.models.metrics import binary_accuracy
from smartai.models.metrics import categorical_accuracy

EPS = sys.float_info.epsilon


def test_binary_accuracy():
    y_true = torch.tensor([0, 1, 0, 0, 1, 0, 0, 1, 1, 0])
    y_pred = torch.tensor([0, 1, 0, 0, 1, 0, 0, 1, 1, 1])
    acc = binary_accuracy(y_true, y_pred)
    assert abs(acc - 0.9) < EPS, 'Accuracy should be 0.9.'


def test_categorical_accuracy():
    pass
