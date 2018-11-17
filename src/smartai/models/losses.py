from collections import namedtuple as _namedtuple
import torch as _torch

__all__ = [
    'TORCH_LOSSES'
]

_torch_losses = {
    c: _torch.nn.modules.loss.__dict__[c] for c in dir(_torch.nn.modules.loss)
    if c.endswith('Loss') and (not c.startswith('_'))
}
locals().update(_torch_losses)

_Torch_Losses = _namedtuple('Torch_Losses', _torch_losses.keys())
TORCH_LOSSES = _Torch_Losses(**_torch_losses)

if __name__ == '__main__':
    print("PyTorch has {} builtin loss Classes.".format(len(_torch_losses)))
