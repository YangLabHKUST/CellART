from typing import Tuple, List

import torch
from torch import nn

def _get_shapes(model: nn.Module,
                channels: int = 3,
                size: Tuple[int, int] = (224, 224)) -> List[Tuple[int, ...]]:
    """Extract shapes of feature maps computed by the model.

    The model must be an nn.Module whose __call__ method returns all feature
    maps when called with an input.
    """
    # save state so we can restore laterD
    state = model.training

    model.eval()
    with torch.no_grad():
        x = torch.empty(1, channels, *size)
        feats = model(x)

    # restore state
    model.train(state)

    if isinstance(feats, torch.Tensor):
        feats = [feats]

    feat_shapes = [f.shape for f in feats]
    return feat_shapes