# -*- coding: utf-8 -*-
# @Time : 2022/7/29 15:51
# @Author : songweinan
# @Software: PyCharm
# 自能成羽翼，何必仰云梯。
# ======================================================================================================================
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import DropPath
from thops.registry import MODELS
from mmcls.models.utils import make_divisible, InvertedResidual
from mmcls.models.utils.se_layer import SELayer
from mmcls.models.backbones import MobileNetV3 as MobileNetV3_


class PaddleInvertedResidual(InvertedResidual):
    """与mmcv的InvertedResidual区别在于无论in_channels是否等于mid_channels都会使用expand_conv"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 kernel_size=3,
                 stride=1,
                 se_cfg=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 drop_path_rate=0.,
                 with_cp=False,
                 init_cfg=None):
        super(InvertedResidual, self).__init__(init_cfg)
        self.with_res_shortcut = (stride == 1 and in_channels == out_channels)
        assert stride in [1, 2]
        self.with_cp = with_cp
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.with_se = se_cfg is not None
        self.with_expand_conv = True
        self.with_cp = in_channels != out_channels or stride != 1
        if self.with_se:
            assert isinstance(se_cfg, dict)

        if self.with_expand_conv:
            self.expand_conv = ConvModule(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        self.depthwise_conv = ConvModule(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=mid_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if self.with_se:
            self.se = SELayer(**se_cfg)
        self.linear_conv = ConvModule(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)


@MODELS.register_module()
class MobileNetV3(MobileNetV3_):
    """
    在MobileNetV3基础上增加全局屏蔽SE模块和宽度缩放
    """
    def __init__(self,
                 widen_factor=0.5,
                 disable_se=False,
                 paddle_style=False,
                 **kwargs):
        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert widen_factor in supported_scale, \
            "supported scale are {} but input scale is {}".format(supported_scale, widen_factor)
        self.widen_factor = widen_factor
        self.disable_se = disable_se
        self.paddle_style = paddle_style
        super(MobileNetV3, self).__init__(**kwargs)

    def _make_layer(self):
        layers = []
        layer_setting = self.arch_settings[self.arch]
        in_channels = make_divisible(16 * self.widen_factor, 8)

        layer = ConvModule(
            in_channels=3,
            out_channels=in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='HSwish'))
        self.add_module('layer0', layer)
        layers.append('layer0')

        for i, params in enumerate(layer_setting):
            (kernel_size, mid_channels, out_channels, with_se, act,
             stride) = params
            mid_channels = make_divisible(mid_channels * self.widen_factor, 8)
            out_channels = make_divisible(out_channels * self.widen_factor, 8)
            with_se = False if self.disable_se else with_se
            if with_se:
                se_cfg = dict(
                    channels=mid_channels,
                    ratio=4,
                    act_cfg=(dict(type='ReLU'),
                             dict(
                                 type='HSigmoid' if not self.paddle_style else 'PaddleHSigmoid',
                                 # type='HSigmoid',
                                 bias=3,
                                 divisor=6,
                                 min_value=0,
                                 max_value=1)))
            else:
                se_cfg = None

            InvertedResidual_ = PaddleInvertedResidual if self.paddle_style else InvertedResidual
            layer = InvertedResidual_(
                in_channels=in_channels,
                out_channels=out_channels,
                mid_channels=mid_channels,
                kernel_size=kernel_size,
                stride=stride,
                se_cfg=se_cfg,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=dict(type=act),
                with_cp=self.with_cp)
            in_channels = out_channels
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, layer)
            layers.append(layer_name)

        # Build the last layer before pooling
        out_channels = 576 if self.arch == 'small' else 960
        out_channels = make_divisible(out_channels * self.widen_factor, 8)
        layer = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='HSwish'))
        layer_name = 'layer{}'.format(len(layer_setting) + 1)
        self.add_module(layer_name, layer)
        layers.append(layer_name)

        return layers
