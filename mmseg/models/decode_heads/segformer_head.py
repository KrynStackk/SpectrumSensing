# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize

class ChannelWiseAttention(nn.Module):
    def __init__(self, input_dim, squeeze_factor=4):
        super(ChannelWiseAttention, self).__init__()
        self.squeeze_dim = input_dim // squeeze_factor
        self.fc1 = nn.Conv2d(input_dim, self.squeeze_dim, kernel_size=1)
        self.fc2 = nn.Conv2d(self.squeeze_dim, input_dim, kernel_size=1)

    def forward(self, x):
        # Global Average Pooling
        gap = F.adaptive_avg_pool2d(x, (1, 1))
        # Conv layers
        squeeze = F.relu(self.fc1(gap))
        excitation = torch.sigmoid(self.fc2(squeeze))
        # Scale input
        output = x * excitation
        return output

@MODELS.register_module()
class SegformerHead(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.conv1x1= ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        self.channel_wise_attention = ChannelWiseAttention(input_dim=self.channels * num_inputs)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.channel_wise_attention(torch.cat(outs, dim=1))
        out = self.conv1x1(out)

        # out = self.cls_seg(out)

        return out
