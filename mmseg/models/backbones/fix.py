# Copyright (c) OpenMMLab. All rights reserved.
# Originally from https://github.com/visual-attention-network/segnext
# Licensed under the Apache License, Version 2.0 (the "License")
import math
import warnings

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule
from mmengine.model.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)

from mmseg.registry import MODELS


class Mlp(BaseModule):
    """Multi Layer Perceptron (MLP) Module.

    Args:
        in_features (int): The dimension of input features.
        hidden_features (int): The dimension of hidden features.
            Defaults: None.
        out_features (int): The dimension of output features.
            Defaults: None.
        act_cfg (dict): Config dict for activation layer in block.
            Default: dict(type='GELU').
        drop (float): The number of dropout rate in MLP block.
            Defaults: 0.0.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            3,
            1,
            1,
            bias=True,
            groups=hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """Forward function."""

        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class StemConv(BaseModule):
    """Stem Block at the beginning of Semantic Branch.

    Args:
        in_channels (int): The dimension of input channels.
        out_channels (int): The dimension of output channels.
        act_cfg (dict): Config dict for activation layer in block.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Defaults: dict(type='SyncBN', requires_grad=True).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)),
            build_norm_layer(norm_cfg, out_channels // 2)[1],
            build_activation_layer(act_cfg),
            nn.Conv2d(
                out_channels // 2,
                out_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)),
            build_norm_layer(norm_cfg, out_channels)[1],
        )

    def forward(self, x):
        """Forward function."""

        x = self.proj(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class MSCAAttention(BaseModule):
    """Modified MSCA Attention Module based on the provided diagram.

    Args:
        channels (int): The dimension of channels.
        kernel_sizes (list): The size of attention kernel. Defaults: [3, 3, 3, 5].
        rates (list): The dilation rates for each branch. Defaults: [3, 5, 7].
        paddings (list): The corresponding padding value in attention module.
            Defaults: [1, 2, 3].
    """

    def __init__(self,
                 channels,
                 kernel_sizes=[3, 3, 3, 5],
                 rates=[3, 5, 7],
                 paddings=[1, 2, 3]):
        super().__init__()
        # First branch with 1x1 -> 3x3 with dilation rate 3
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Conv2d(channels, channels, kernel_size=kernel_sizes[0], 
                      padding=paddings[0], dilation=rates[0], groups=channels)
        )
        # Second branch with 1x1 -> 3x3 with dilation rate 5
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Conv2d(channels, channels, kernel_size=kernel_sizes[1], 
                      padding=paddings[1], dilation=rates[1], groups=channels)
        )
        # Third branch with 1x1 -> 3x3 with dilation rate 7
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Conv2d(channels, channels, kernel_size=kernel_sizes[2], 
                      padding=paddings[2], dilation=rates[2], groups=channels)
        )
        # Fourth branch with direct 5x5 convolution
        self.branch4 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)

        # Mixing branch outputs
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        """Forward function."""
        # Clone the input for attention
        u = x.clone()

        # Apply each branch and sum the outputs
        attn = self.branch1(x) + self.branch2(x) + self.branch3(x) + self.branch4(x)

        # Final 1x1 convolution to mix the channels
        attn = self.conv1x1(attn)

        # Multiply by the original input for attention effect
        x = attn * u

        return x



class MSCASpatialAttention(BaseModule):
    """Modified Spatial Attention Module in Multi-Scale Convolutional Attention Module (MSCA).

    Args:
        in_channels (int): The dimension of channels.
        attention_kernel_sizes (list): The size of attention kernel. Defaults: [3, 3, 3, 5].
        attention_kernel_paddings (list): The corresponding padding values for each kernel size.
        dilation_rates (list): The dilation rates for each branch. Defaults: [3, 5, 7].
        act_cfg (dict): Config dict for activation layer in block.
            Default: dict(type='GELU').
    """

    def __init__(self,
                 in_channels,
                 attention_kernel_sizes=[3, 3, 3, 5],
                 attention_kernel_paddings=[1, 2, 3, 2],
                 dilation_rates=[3, 5, 7],
                 act_cfg=dict(type='GELU')):
        super().__init__()
        # Initialize the projection layer
        self.proj_1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.activation = build_activation_layer(act_cfg)

        # Attention branches
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=attention_kernel_sizes[0],
                      padding=attention_kernel_paddings[0], dilation=dilation_rates[0], groups=in_channels)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=attention_kernel_sizes[1],
                      padding=attention_kernel_paddings[1], dilation=dilation_rates[1], groups=in_channels)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=attention_kernel_sizes[2],
                      padding=attention_kernel_paddings[2], dilation=dilation_rates[2], groups=in_channels)
        )
        self.branch4 = nn.Conv2d(in_channels, in_channels, kernel_size=attention_kernel_sizes[3], 
                                 padding=attention_kernel_paddings[3], groups=in_channels)

        # Final 1x1 convolution after mixing the branches
        self.proj_2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        """Forward function."""

        # Store the original input for skip connection
        shortcut = x.clone()

        # Apply the first projection and activation
        x = self.proj_1(x)
        x = self.activation(x)

        # Apply the attention branches
        x = self.branch1(x) + self.branch2(x) + self.branch3(x) + self.branch4(x)

        # Final projection and skip connection
        x = self.proj_2(x)
        x = x + shortcut  # Skip connection to retain input information

        return x



class MSCABlock(BaseModule):
    """Modified Multi-Scale Convolutional Attention Block.

    This block uses both spatial and channel attention with different dilation rates and kernel sizes, based on the previous modified classes.

    Args:
        channels (int): The dimension of channels.
        attention_kernel_sizes (list): The size of attention kernel.
            Defaults: [3, 3, 3, 5].
        attention_kernel_paddings (list): Padding for each attention kernel.
            Defaults: [1, 2, 3, 2].
        dilation_rates (list): Dilation rates for each attention branch.
            Defaults: [3, 5, 7].
        mlp_ratio (float): The ratio of hidden feature dimension in MLP to the input dimension.
            Defaults: 4.0.
        drop (float): The dropout rate.
            Defaults: 0.0.
        drop_path (float): The ratio of drop paths for stochastic depth.
            Defaults: 0.0.
        act_cfg (dict): Configuration for activation layer.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Defaults: dict(type='SyncBN', requires_grad=True).
    """

    def __init__(self,
                 channels,
                 attention_kernel_sizes=[3, 3, 3, 5],
                 attention_kernel_paddings=[1, 2, 3, 2],
                 dilation_rates=[3, 5, 7],
                 mlp_ratio=4.0,
                 drop=0.0,
                 drop_path=0.0,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        # Normalization layers before attention and MLP layers
        self.norm1 = build_norm_layer(norm_cfg, channels)[1]
        self.norm2 = build_norm_layer(norm_cfg, channels)[1]

        # Multi-Scale Convolutional Spatial Attention (updated MSCA)
        self.attn = MSCASpatialAttention(
            in_channels=channels,
            attention_kernel_sizes=attention_kernel_sizes,
            attention_kernel_paddings=attention_kernel_paddings,
            dilation_rates=dilation_rates,
            act_cfg=act_cfg
        )

        # Drop path for stochastic depth regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Multi-Layer Perceptron (MLP) with hidden channels based on the ratio
        mlp_hidden_channels = int(channels * mlp_ratio)
        self.mlp = Mlp(
            in_features=channels,
            hidden_features=mlp_hidden_channels,
            act_cfg=act_cfg,
            drop=drop
        )

        # Layer scales for weighting
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(channels), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(channels), requires_grad=True)

    def forward(self, x, H, W):
        """Forward function."""
        B, N, C = x.shape  # Batch size, sequence length, channels

        # Reshape to (B, C, H, W) for convolutions
        x = x.permute(0, 2, 1).view(B, C, H, W)

        # Apply MSCA spatial attention
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))

        # Apply MLP block
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))

        # Reshape back to (B, N, C)
        x = x.view(B, C, N).permute(0, 2, 1)

        return x


class OverlapPatchEmbed(BaseModule):
    """Modified Image to Patch Embedding.

    Args:
        patch_size (int): The patch size. Defaults: 7.
        stride (int): Stride of the convolutional layer. Defaults: 4.
        in_channels (int): The number of input channels. Defaults: 3.
        embed_dims (int): The dimensions of embedding. Defaults: 768.
        norm_cfg (dict): Config for normalization layer. Defaults: dict(type='SyncBN', requires_grad=True).
        kernel_size (int): The kernel size for the embedding convolution. Defaults: 7.
        dilation (int): The dilation rate for convolution. Defaults: 1.
    """

    def __init__(self,
                 patch_size=7,
                 stride=4,
                 in_channels=3,
                 embed_dim=768,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 kernel_size=7,
                 dilation=1):
        super().__init__()

        # Convolution for patch embedding with optional dilation
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size // 2),
            dilation=dilation)

        # Normalization layer for the output embedding
        self.norm = build_norm_layer(norm_cfg, embed_dim)[1]

    def forward(self, x):
        """Forward function for patch embedding."""

        # Apply convolution to create the patch embeddings
        x = self.proj(x)

        # Get the spatial dimensions of the output
        _, _, H, W = x.shape

        # Apply normalization
        x = self.norm(x)

        # Flatten the output into patch sequences for the next layers
        x = x.flatten(2).transpose(1, 2)  # (B, C, H*W) -> (B, H*W, C)

        return x, H, W



@MODELS.register_module()
class Thinh(BaseModule):
    """Modified Multi-Scale Convolutional Attention Network (MSCAN) backbone.

    This backbone is based on multi-scale convolutional attention blocks and flexible patch embeddings.

    Args:
        in_channels (int): The number of input channels. Defaults: 3.
        embed_dims (list[int]): Embedding dimension for each stage. Defaults: [64, 128, 256, 512].
        mlp_ratios (list[int]): Ratio of mlp hidden dim to embedding dim. Defaults: [4, 4, 4, 4].
        drop_rate (float): Dropout rate. Defaults: 0.
        drop_path_rate (float): Stochastic depth rate. Defaults: 0.
        depths (list[int]): Depths of each stage (number of blocks per stage). Default: [3, 4, 6, 3].
        num_stages (int): Number of stages in the network. Default: 4.
        attention_kernel_sizes (list): Size of attention kernel in the MSCA module. Defaults: [3, 3, 3, 5].
        attention_kernel_paddings (list): Padding sizes for the attention kernels. Defaults: [1, 2, 3, 2].
        dilation_rates (list): Dilation rates for the attention branches. Defaults: [3, 5, 7].
        norm_cfg (dict): Config for normalization layers. Defaults: dict(type='SyncBN', requires_grad=True).
        pretrained (str, optional): Path to pretrained weights. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict. Default: None.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 depths=[3, 4, 6, 3],
                 num_stages=4,
                 attention_kernel_sizes=[3, 3, 3, 5],
                 attention_kernel_paddings=[1, 2, 3, 2],
                 dilation_rates=[3, 5, 7],
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.depths = depths
        self.num_stages = num_stages

        # Drop path probabilities for stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        # Build the stages
        for i in range(num_stages):
            # Modify in_channels to 24 if the input has 24 channels
            if i == 0:
                patch_embed = StemConv(in_channels=24, out_channels=embed_dims[0], norm_cfg=norm_cfg)
            else:
                patch_embed = OverlapPatchEmbed(
                    patch_size=3 if i != 0 else 7,
                    stride=2,
                    in_channels=embed_dims[i - 1],
                    embed_dim=embed_dims[i],
                    norm_cfg=norm_cfg,
                    kernel_size=3,
                    dilation=1
                )

            # Multi-Scale Convolutional Attention blocks for each stage
            block = nn.ModuleList([
                MSCABlock(
                    channels=embed_dims[i],
                    attention_kernel_sizes=attention_kernel_sizes,
                    attention_kernel_paddings=attention_kernel_paddings,
                    dilation_rates=dilation_rates,
                    mlp_ratio=mlp_ratios[i],
                    drop=drop_rate,
                    drop_path=dpr[cur + j],
                    act_cfg=dict(type='GELU'),
                    norm_cfg=norm_cfg) for j in range(depths[i])
            ])

            # Normalization for the output of each stage
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f'patch_embed{i + 1}', patch_embed)
            setattr(self, f'block{i + 1}', block)
            setattr(self, f'norm{i + 1}', norm)

    def init_weights(self):
        """Initialize the weights of MSCAN."""
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super().init_weights()

    def forward(self, x):
        """Forward pass through MSCAN backbone."""
        B = x.shape[0]  # Batch size
        outs = []

        # Forward through each stage
        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            block = getattr(self, f'block{i + 1}')
            norm = getattr(self, f'norm{i + 1}')

            # Apply the patch embedding
            x, H, W = patch_embed(x)

            # Apply each block in the stage
            for blk in block:
                x = blk(x, H, W)

            # Apply normalization
            x = norm(x)

            # Reshape for the next stage
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            # Append the output of the stage
            outs.append(x)

        return outs
