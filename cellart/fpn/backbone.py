from typing import Tuple, Optional, Type

from torch import nn

from .containers import (Parallel, SequentialMultiInputMultiOutput,
                        SequentialMultiOutput)
from .layers import (Sum, SplitTensor)

class ResNetFeatureMapsExtractor(nn.Module):
    def __init__(self, model: nn.Module, mode: Optional[str] = None):
        super().__init__()
        self.mode = mode
        # yapf: disable
        stem = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool
        )
        layers = [
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        ]
        if mode == 'fusion':
            # allow each layer to take in multiple inputs - summming them
            # before passing them through. This allows each layer to take
            # in feature maps from multiple backbones.
            multi_input_layers = [
                nn.Sequential(Sum(), layer) for layer in layers
            ]
            self.m = SequentialMultiInputMultiOutput(
                stem,
                *multi_input_layers,
                Sum()
            )
        else:
            self.m = SequentialMultiOutput(stem, *layers)

    def forward(self, x):
        if self.mode != 'fusion':
            return self.m(x)
        x, in_feats = x
        return self.m((x, *in_feats))