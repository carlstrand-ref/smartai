from collections import namedtuple
import torch

__all__ = [
    'TORCH_OPTIMIZER'
]

_torch_optimizers = {
    c: torch.optim.__dict__[c] for c in dir(torch.optim)
    if c[0].isupper() and c != 'Optimizer'
}
locals().update(_torch_optimizers)
print("PyTorch has {} builtin optimizer Classes.".format(len(_torch_optimizers)))

_Torch_Optimizer = namedtuple('Torch_Optimizer', _torch_optimizers.keys())
TORCH_OPTIMIZER = _Torch_Optimizer(**_torch_optimizers)
