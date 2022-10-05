# @Time : 2022/7/29 17:39
# @Author : songweinan
# @Software: PyCharm
# 自能成羽翼，何必仰云梯。
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.utils.se_layer import SELayer
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from thops.registry import MODELS


class RSELayer(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 shortcut=True,
                 init_cfg=None):
        super(RSELayer, self).__init__(init_cfg)
        self.out_channels = out_channels
        self.in_conv = ConvModule(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
            act_cfg=None)
        se_cfg = dict(
            channels=self.out_channels,
            squeeze_channels=self.out_channels // 4,
            ratio=4,
            act_cfg=(dict(type='ReLU'),
                     dict(
                         type='thops.PaddleHSigmoid',
                         bias=3,
                         divisor=6,
                         min_value=0,
                         max_value=1)))
        self.se = SELayer(**se_cfg)
        self.shortcut = shortcut

    def forward(self, x):
        x = self.in_conv(x)
        if self.shortcut:
            out = x + self.se(x)
        else:
            out = self.se(x)
        return out


@MODELS.register_module()
class RSEFPN(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 init_cfg=dict(
                     type='Xavier',
                     layer=['Conv2d', 'ConvTranspose2d'],
                     distribution='uniform')):
        super(RSEFPN, self).__init__(init_cfg=init_cfg)
        assert len(in_channels) == 4
        assert isinstance(out_channels, int)
        self.in_channels = in_channels
        self.ins_conv = nn.ModuleList()
        self.inp_conv = nn.ModuleList()

        for i in range(len(in_channels)):
            self.ins_conv.append(
                RSELayer(
                    in_channels[i],
                    out_channels,
                    kernel_size=1,
                    shortcut=shortcut))
            self.inp_conv.append(
                RSELayer(
                    out_channels,
                    out_channels // 4,
                    kernel_size=3,
                    shortcut=shortcut))

    def forward(self, x):
        assert len(x) == len(self.in_channels)
        c2, c3, c4, c5 = x

        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)

        out4 = in4 + F.upsample(in5, scale_factor=2, mode='nearest')  # 1/16
        out3 = in3 + F.upsample(out4, scale_factor=2, mode='nearest')  # 1/8
        out2 = in2 + F.upsample(out3, scale_factor=2, mode='nearest')  # 1/4

        p5 = self.inp_conv[3](in5)
        p4 = self.inp_conv[2](out4)
        p3 = self.inp_conv[1](out3)
        p2 = self.inp_conv[0](out2)

        p5 = F.upsample(p5, scale_factor=8, mode='nearest')
        p4 = F.upsample(p4, scale_factor=4, mode='nearest')
        p3 = F.upsample(p3, scale_factor=2, mode='nearest')

        fuse = torch.concat([p5, p4, p3, p2], axis=1)
        return fuse
