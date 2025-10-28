# https://github.com/AdeelH/pytorch-fpn/blob/master/fpn/factory.py
from typing import Tuple, Optional
import torch
from torch import nn
import torchvision as tv
from .backbone import ResNetFeatureMapsExtractor
from .layers import Interpolate, SelectOne
from .fpn import FPN, PanopticFPN
from .utils import _get_shapes

def make_spot_fpn_resnet(name: str = 'resnet18',
                         fpn_type: str = 'fpn',
                         out_size: Tuple[int, int] = (224, 224),
                         fpn_channels: int = 256,
                         num_classes: int = 1000,
                         in_channels: int = 500):
    assert in_channels > 0
    assert num_classes > 0

    resnet = tv.models.resnet.__dict__[name](pretrained=False)

    old_conv = resnet.conv1
    old_conv_args = {
        'out_channels': old_conv.out_channels,
        'kernel_size': old_conv.kernel_size,
        'stride': old_conv.stride,
        'padding': old_conv.padding,
        'dilation': old_conv.dilation,
        'groups': old_conv.groups,
        'bias': old_conv.bias
    }
    # just replace the first conv layer
    new_conv = nn.Conv2d(in_channels=in_channels, **old_conv_args)
    resnet.conv1 = new_conv
    backbone = ResNetFeatureMapsExtractor(resnet)

    feat_shapes = _get_shapes(backbone, channels=in_channels, size=out_size)
    if fpn_type == 'fpn':
        fpn = nn.Sequential(
            FPN(feat_shapes,
                hidden_channels=fpn_channels,
                out_channels=num_classes),
            SelectOne(idx=0))
    elif fpn_type == 'panoptic':
        fpn = PanopticFPN(
            feat_shapes,
            hidden_channels=fpn_channels,
            out_channels=num_classes)
    else:
        raise NotImplementedError()

    # yapf: disable
    model = nn.Sequential(
        backbone,
        fpn,
        Interpolate(size=out_size))
    # yapf: enable
    return model