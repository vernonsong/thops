# -*- coding: utf-8 -*-
# @Time : 2022/8/8 08:20
# @Author : songweinan
# @Software: PyCharm
# 自能成羽翼，何必仰云梯。
# ======================================================================================================================
import torch.nn as nn
from mmengine.model import BaseModule
from mmcv.cnn.bricks import ConvModule
from mmcls.models.utils.se_layer import SELayer
from thops.registry import MODELS


class DepthwiseSeparable(BaseModule):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 num_groups,
                 scale,
                 kernel_size=3,
                 stride=1,
                 with_se=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='HSwish'),
                 init_cfg=None):
        super(DepthwiseSeparable, self).__init__(init_cfg)
        self.with_res_shortcut = (stride == 1 and in_channels == out_channels)
        # assert stride in [1, 2]
        self.with_se = with_se


        self.depthwise_conv = ConvModule(
            in_channels=in_channels,
            out_channels=int(mid_channels * scale),
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=int(num_groups * scale),
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if self.with_se:
            se_cfg = dict(
                channels=int(mid_channels * scale),
                ratio=4,
                act_cfg=(dict(type='ReLU'),
                         dict(
                             type='HSigmoid',
                             bias=3,
                             divisor=6,
                             min_value=0,
                             max_value=1)))

            self.se = SELayer(**se_cfg)
        self.linear_conv = ConvModule(
            in_channels=int(mid_channels * scale),
            out_channels=int(out_channels * scale),
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        x = self.depthwise_conv(x)
        if self.with_se:
            x = self.se(x)
        out = self.linear_conv(x)
        return out


@MODELS.register_module()
class MobileNetEnhanceBackbone(BaseModule):
    def __init__(self,
                 in_channels=3,
                 scale=0.5,
                 last_conv_stride=1,
                 last_pool_type='avg',
                 init_cfg=None):
        super(MobileNetEnhanceBackbone, self).__init__(init_cfg)
        self.scale = scale
        self.block_list = []
        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=int(32 * scale),
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='HSwish'),
            bias=False)

        conv2_1 = DepthwiseSeparable(
                in_channels=int(32 * scale),
                mid_channels=32,
                out_channels=64,
                num_groups=32,
                stride=1,
                scale=scale)

        self.block_list.append(conv2_1)

        conv2_2 = DepthwiseSeparable(
            in_channels=int(64 * scale),
            mid_channels=64,
            out_channels=128,
            num_groups=64,
            stride=1,
            scale=scale)

        self.block_list.append(conv2_2)

        conv3_1 = DepthwiseSeparable(
            in_channels=int(128 * scale),
            mid_channels=128,
            out_channels=128,
            num_groups=128,
            stride=1,
            scale=scale)
        self.block_list.append(conv3_1)

        conv3_2 = DepthwiseSeparable(
            in_channels=int(128 * scale),
            mid_channels=128,
            out_channels=256,
            num_groups=128,
            stride=(2, 1),
            scale=scale)
        self.block_list.append(conv3_2)

        conv4_1 = DepthwiseSeparable(
            in_channels=int(256 * scale),
            mid_channels=256,
            out_channels=256,
            num_groups=256,
            stride=1,
            scale=scale)
        self.block_list.append(conv4_1)

        conv4_2 = DepthwiseSeparable(
            in_channels=int(256 * scale),
            mid_channels=256,
            out_channels=512,
            num_groups=256,
            stride=(2, 1),
            scale=scale)
        self.block_list.append(conv4_2)

        for _ in range(5):
            conv5 = DepthwiseSeparable(
                in_channels=int(512 * scale),
                mid_channels=512,
                out_channels=512,
                num_groups=512,
                stride=1,
                kernel_size=5,
                scale=scale,
                with_se=False)
            self.block_list.append(conv5)

        conv5_6 = DepthwiseSeparable(
            in_channels=int(512 * scale),
            mid_channels=512,
            out_channels=1024,
            num_groups=512,
            stride=(2, 1),
            kernel_size=5,
            scale=scale,
            with_se=True)
        self.block_list.append(conv5_6)

        conv6 = DepthwiseSeparable(
            in_channels=int(1024 * scale),
            mid_channels=1024,
            out_channels=1024,
            num_groups=1024,
            stride=last_conv_stride,
            kernel_size=5,
            with_se=True,
            scale=scale)
        self.block_list.append(conv6)

        self.block_list = nn.Sequential(*self.block_list)
        if last_pool_type == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=2,stride=2, padding=0)
        else:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.out_channels = int(1024 * scale)

    def forward(self, inputs):
        y = self.conv1(inputs)
        y = self.block_list(y)
        y = self.pool(y)
        return y



