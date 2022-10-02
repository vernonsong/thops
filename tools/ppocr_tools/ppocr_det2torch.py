# @Time : 2022/7/31 13:43
# @Author : songweinan
# @Software: PyCharm
# 自能成羽翼，何必仰云梯。
# ======================================================================================================================
import mmcv
import numpy as np
import paddle.fluid as fluid
import torch
from mmengine import Config
from mmengine.dataset.base_dataset import Compose
from mmengine.registry import MODELS
from paddle import inference

from thops.utils import register_all_modules

STATE_MAP = {
    'backbone.layer0.conv.weight': 'backbone.conv.conv.weight',
    'backbone.layer0.bn.weight': 'backbone.conv.bn.weight',
    'backbone.layer0.bn.bias': 'backbone.conv.bn.bias',
    'backbone.layer0.bn.running_mean': 'backbone.conv.bn._mean',
    'backbone.layer0.bn.running_var': 'backbone.conv.bn._variance',
    'backbone.layer1.expand_conv.conv.weight':
    'backbone.stage0.0.expand_conv.conv.weight',
    'backbone.layer1.expand_conv.bn.weight':
    'backbone.stage0.0.expand_conv.bn.weight',
    'backbone.layer1.expand_conv.bn.bias':
    'backbone.stage0.0.expand_conv.bn.bias',
    'backbone.layer1.expand_conv.bn.running_mean':
    'backbone.stage0.0.expand_conv.bn._mean',
    'backbone.layer1.expand_conv.bn.running_var':
    'backbone.stage0.0.expand_conv.bn._variance',
    'backbone.layer1.depthwise_conv.conv.weight':
    'backbone.stage0.0.bottleneck_conv.conv.weight',
    'backbone.layer1.depthwise_conv.bn.weight':
    'backbone.stage0.0.bottleneck_conv.bn.weight',
    'backbone.layer1.depthwise_conv.bn.bias':
    'backbone.stage0.0.bottleneck_conv.bn.bias',
    'backbone.layer1.depthwise_conv.bn.running_mean':
    'backbone.stage0.0.bottleneck_conv.bn._mean',
    'backbone.layer1.depthwise_conv.bn.running_var':
    'backbone.stage0.0.bottleneck_conv.bn._variance',
    'backbone.layer1.linear_conv.conv.weight':
    'backbone.stage0.0.linear_conv.conv.weight',
    'backbone.layer1.linear_conv.bn.weight':
    'backbone.stage0.0.linear_conv.bn.weight',
    'backbone.layer1.linear_conv.bn.bias':
    'backbone.stage0.0.linear_conv.bn.bias',
    'backbone.layer1.linear_conv.bn.running_mean':
    'backbone.stage0.0.linear_conv.bn._mean',
    'backbone.layer1.linear_conv.bn.running_var':
    'backbone.stage0.0.linear_conv.bn._variance',
    'backbone.layer2.expand_conv.conv.weight':
    'backbone.stage0.1.expand_conv.conv.weight',
    'backbone.layer2.expand_conv.bn.weight':
    'backbone.stage0.1.expand_conv.bn.weight',
    'backbone.layer2.expand_conv.bn.bias':
    'backbone.stage0.1.expand_conv.bn.bias',
    'backbone.layer2.expand_conv.bn.running_mean':
    'backbone.stage0.1.expand_conv.bn._mean',
    'backbone.layer2.expand_conv.bn.running_var':
    'backbone.stage0.1.expand_conv.bn._variance',
    'backbone.layer2.depthwise_conv.conv.weight':
    'backbone.stage0.1.bottleneck_conv.conv.weight',
    'backbone.layer2.depthwise_conv.bn.weight':
    'backbone.stage0.1.bottleneck_conv.bn.weight',
    'backbone.layer2.depthwise_conv.bn.bias':
    'backbone.stage0.1.bottleneck_conv.bn.bias',
    'backbone.layer2.depthwise_conv.bn.running_mean':
    'backbone.stage0.1.bottleneck_conv.bn._mean',
    'backbone.layer2.depthwise_conv.bn.running_var':
    'backbone.stage0.1.bottleneck_conv.bn._variance',
    'backbone.layer2.linear_conv.conv.weight':
    'backbone.stage0.1.linear_conv.conv.weight',
    'backbone.layer2.linear_conv.bn.weight':
    'backbone.stage0.1.linear_conv.bn.weight',
    'backbone.layer2.linear_conv.bn.bias':
    'backbone.stage0.1.linear_conv.bn.bias',
    'backbone.layer2.linear_conv.bn.running_mean':
    'backbone.stage0.1.linear_conv.bn._mean',
    'backbone.layer2.linear_conv.bn.running_var':
    'backbone.stage0.1.linear_conv.bn._variance',
    'backbone.layer3.expand_conv.conv.weight':
    'backbone.stage0.2.expand_conv.conv.weight',
    'backbone.layer3.expand_conv.bn.weight':
    'backbone.stage0.2.expand_conv.bn.weight',
    'backbone.layer3.expand_conv.bn.bias':
    'backbone.stage0.2.expand_conv.bn.bias',
    'backbone.layer3.expand_conv.bn.running_mean':
    'backbone.stage0.2.expand_conv.bn._mean',
    'backbone.layer3.expand_conv.bn.running_var':
    'backbone.stage0.2.expand_conv.bn._variance',
    'backbone.layer3.depthwise_conv.conv.weight':
    'backbone.stage0.2.bottleneck_conv.conv.weight',
    'backbone.layer3.depthwise_conv.bn.weight':
    'backbone.stage0.2.bottleneck_conv.bn.weight',
    'backbone.layer3.depthwise_conv.bn.bias':
    'backbone.stage0.2.bottleneck_conv.bn.bias',
    'backbone.layer3.depthwise_conv.bn.running_mean':
    'backbone.stage0.2.bottleneck_conv.bn._mean',
    'backbone.layer3.depthwise_conv.bn.running_var':
    'backbone.stage0.2.bottleneck_conv.bn._variance',
    'backbone.layer3.linear_conv.conv.weight':
    'backbone.stage0.2.linear_conv.conv.weight',
    'backbone.layer3.linear_conv.bn.weight':
    'backbone.stage0.2.linear_conv.bn.weight',
    'backbone.layer3.linear_conv.bn.bias':
    'backbone.stage0.2.linear_conv.bn.bias',
    'backbone.layer3.linear_conv.bn.running_mean':
    'backbone.stage0.2.linear_conv.bn._mean',
    'backbone.layer3.linear_conv.bn.running_var':
    'backbone.stage0.2.linear_conv.bn._variance',
    'backbone.layer4.expand_conv.conv.weight':
    'backbone.stage1.0.expand_conv.conv.weight',
    'backbone.layer4.expand_conv.bn.weight':
    'backbone.stage1.0.expand_conv.bn.weight',
    'backbone.layer4.expand_conv.bn.bias':
    'backbone.stage1.0.expand_conv.bn.bias',
    'backbone.layer4.expand_conv.bn.running_mean':
    'backbone.stage1.0.expand_conv.bn._mean',
    'backbone.layer4.expand_conv.bn.running_var':
    'backbone.stage1.0.expand_conv.bn._variance',
    'backbone.layer4.depthwise_conv.conv.weight':
    'backbone.stage1.0.bottleneck_conv.conv.weight',
    'backbone.layer4.depthwise_conv.bn.weight':
    'backbone.stage1.0.bottleneck_conv.bn.weight',
    'backbone.layer4.depthwise_conv.bn.bias':
    'backbone.stage1.0.bottleneck_conv.bn.bias',
    'backbone.layer4.depthwise_conv.bn.running_mean':
    'backbone.stage1.0.bottleneck_conv.bn._mean',
    'backbone.layer4.depthwise_conv.bn.running_var':
    'backbone.stage1.0.bottleneck_conv.bn._variance',
    'backbone.layer4.linear_conv.conv.weight':
    'backbone.stage1.0.linear_conv.conv.weight',
    'backbone.layer4.linear_conv.bn.weight':
    'backbone.stage1.0.linear_conv.bn.weight',
    'backbone.layer4.linear_conv.bn.bias':
    'backbone.stage1.0.linear_conv.bn.bias',
    'backbone.layer4.linear_conv.bn.running_mean':
    'backbone.stage1.0.linear_conv.bn._mean',
    'backbone.layer4.linear_conv.bn.running_var':
    'backbone.stage1.0.linear_conv.bn._variance',
    'backbone.layer5.expand_conv.conv.weight':
    'backbone.stage1.1.expand_conv.conv.weight',
    'backbone.layer5.expand_conv.bn.weight':
    'backbone.stage1.1.expand_conv.bn.weight',
    'backbone.layer5.expand_conv.bn.bias':
    'backbone.stage1.1.expand_conv.bn.bias',
    'backbone.layer5.expand_conv.bn.running_mean':
    'backbone.stage1.1.expand_conv.bn._mean',
    'backbone.layer5.expand_conv.bn.running_var':
    'backbone.stage1.1.expand_conv.bn._variance',
    'backbone.layer5.depthwise_conv.conv.weight':
    'backbone.stage1.1.bottleneck_conv.conv.weight',
    'backbone.layer5.depthwise_conv.bn.weight':
    'backbone.stage1.1.bottleneck_conv.bn.weight',
    'backbone.layer5.depthwise_conv.bn.bias':
    'backbone.stage1.1.bottleneck_conv.bn.bias',
    'backbone.layer5.depthwise_conv.bn.running_mean':
    'backbone.stage1.1.bottleneck_conv.bn._mean',
    'backbone.layer5.depthwise_conv.bn.running_var':
    'backbone.stage1.1.bottleneck_conv.bn._variance',
    'backbone.layer5.linear_conv.conv.weight':
    'backbone.stage1.1.linear_conv.conv.weight',
    'backbone.layer5.linear_conv.bn.weight':
    'backbone.stage1.1.linear_conv.bn.weight',
    'backbone.layer5.linear_conv.bn.bias':
    'backbone.stage1.1.linear_conv.bn.bias',
    'backbone.layer5.linear_conv.bn.running_mean':
    'backbone.stage1.1.linear_conv.bn._mean',
    'backbone.layer5.linear_conv.bn.running_var':
    'backbone.stage1.1.linear_conv.bn._variance',
    'backbone.layer6.expand_conv.conv.weight':
    'backbone.stage1.2.expand_conv.conv.weight',
    'backbone.layer6.expand_conv.bn.weight':
    'backbone.stage1.2.expand_conv.bn.weight',
    'backbone.layer6.expand_conv.bn.bias':
    'backbone.stage1.2.expand_conv.bn.bias',
    'backbone.layer6.expand_conv.bn.running_mean':
    'backbone.stage1.2.expand_conv.bn._mean',
    'backbone.layer6.expand_conv.bn.running_var':
    'backbone.stage1.2.expand_conv.bn._variance',
    'backbone.layer6.depthwise_conv.conv.weight':
    'backbone.stage1.2.bottleneck_conv.conv.weight',
    'backbone.layer6.depthwise_conv.bn.weight':
    'backbone.stage1.2.bottleneck_conv.bn.weight',
    'backbone.layer6.depthwise_conv.bn.bias':
    'backbone.stage1.2.bottleneck_conv.bn.bias',
    'backbone.layer6.depthwise_conv.bn.running_mean':
    'backbone.stage1.2.bottleneck_conv.bn._mean',
    'backbone.layer6.depthwise_conv.bn.running_var':
    'backbone.stage1.2.bottleneck_conv.bn._variance',
    'backbone.layer6.linear_conv.conv.weight':
    'backbone.stage1.2.linear_conv.conv.weight',
    'backbone.layer6.linear_conv.bn.weight':
    'backbone.stage1.2.linear_conv.bn.weight',
    'backbone.layer6.linear_conv.bn.bias':
    'backbone.stage1.2.linear_conv.bn.bias',
    'backbone.layer6.linear_conv.bn.running_mean':
    'backbone.stage1.2.linear_conv.bn._mean',
    'backbone.layer6.linear_conv.bn.running_var':
    'backbone.stage1.2.linear_conv.bn._variance',
    'backbone.layer7.expand_conv.conv.weight':
    'backbone.stage2.0.expand_conv.conv.weight',
    'backbone.layer7.expand_conv.bn.weight':
    'backbone.stage2.0.expand_conv.bn.weight',
    'backbone.layer7.expand_conv.bn.bias':
    'backbone.stage2.0.expand_conv.bn.bias',
    'backbone.layer7.expand_conv.bn.running_mean':
    'backbone.stage2.0.expand_conv.bn._mean',
    'backbone.layer7.expand_conv.bn.running_var':
    'backbone.stage2.0.expand_conv.bn._variance',
    'backbone.layer7.depthwise_conv.conv.weight':
    'backbone.stage2.0.bottleneck_conv.conv.weight',
    'backbone.layer7.depthwise_conv.bn.weight':
    'backbone.stage2.0.bottleneck_conv.bn.weight',
    'backbone.layer7.depthwise_conv.bn.bias':
    'backbone.stage2.0.bottleneck_conv.bn.bias',
    'backbone.layer7.depthwise_conv.bn.running_mean':
    'backbone.stage2.0.bottleneck_conv.bn._mean',
    'backbone.layer7.depthwise_conv.bn.running_var':
    'backbone.stage2.0.bottleneck_conv.bn._variance',
    'backbone.layer7.linear_conv.conv.weight':
    'backbone.stage2.0.linear_conv.conv.weight',
    'backbone.layer7.linear_conv.bn.weight':
    'backbone.stage2.0.linear_conv.bn.weight',
    'backbone.layer7.linear_conv.bn.bias':
    'backbone.stage2.0.linear_conv.bn.bias',
    'backbone.layer7.linear_conv.bn.running_mean':
    'backbone.stage2.0.linear_conv.bn._mean',
    'backbone.layer7.linear_conv.bn.running_var':
    'backbone.stage2.0.linear_conv.bn._variance',
    'backbone.layer8.expand_conv.conv.weight':
    'backbone.stage2.1.expand_conv.conv.weight',
    'backbone.layer8.expand_conv.bn.weight':
    'backbone.stage2.1.expand_conv.bn.weight',
    'backbone.layer8.expand_conv.bn.bias':
    'backbone.stage2.1.expand_conv.bn.bias',
    'backbone.layer8.expand_conv.bn.running_mean':
    'backbone.stage2.1.expand_conv.bn._mean',
    'backbone.layer8.expand_conv.bn.running_var':
    'backbone.stage2.1.expand_conv.bn._variance',
    'backbone.layer8.depthwise_conv.conv.weight':
    'backbone.stage2.1.bottleneck_conv.conv.weight',
    'backbone.layer8.depthwise_conv.bn.weight':
    'backbone.stage2.1.bottleneck_conv.bn.weight',
    'backbone.layer8.depthwise_conv.bn.bias':
    'backbone.stage2.1.bottleneck_conv.bn.bias',
    'backbone.layer8.depthwise_conv.bn.running_mean':
    'backbone.stage2.1.bottleneck_conv.bn._mean',
    'backbone.layer8.depthwise_conv.bn.running_var':
    'backbone.stage2.1.bottleneck_conv.bn._variance',
    'backbone.layer8.linear_conv.conv.weight':
    'backbone.stage2.1.linear_conv.conv.weight',
    'backbone.layer8.linear_conv.bn.weight':
    'backbone.stage2.1.linear_conv.bn.weight',
    'backbone.layer8.linear_conv.bn.bias':
    'backbone.stage2.1.linear_conv.bn.bias',
    'backbone.layer8.linear_conv.bn.running_mean':
    'backbone.stage2.1.linear_conv.bn._mean',
    'backbone.layer8.linear_conv.bn.running_var':
    'backbone.stage2.1.linear_conv.bn._variance',
    'backbone.layer9.expand_conv.conv.weight':
    'backbone.stage2.2.expand_conv.conv.weight',
    'backbone.layer9.expand_conv.bn.weight':
    'backbone.stage2.2.expand_conv.bn.weight',
    'backbone.layer9.expand_conv.bn.bias':
    'backbone.stage2.2.expand_conv.bn.bias',
    'backbone.layer9.expand_conv.bn.running_mean':
    'backbone.stage2.2.expand_conv.bn._mean',
    'backbone.layer9.expand_conv.bn.running_var':
    'backbone.stage2.2.expand_conv.bn._variance',
    'backbone.layer9.depthwise_conv.conv.weight':
    'backbone.stage2.2.bottleneck_conv.conv.weight',
    'backbone.layer9.depthwise_conv.bn.weight':
    'backbone.stage2.2.bottleneck_conv.bn.weight',
    'backbone.layer9.depthwise_conv.bn.bias':
    'backbone.stage2.2.bottleneck_conv.bn.bias',
    'backbone.layer9.depthwise_conv.bn.running_mean':
    'backbone.stage2.2.bottleneck_conv.bn._mean',
    'backbone.layer9.depthwise_conv.bn.running_var':
    'backbone.stage2.2.bottleneck_conv.bn._variance',
    'backbone.layer9.linear_conv.conv.weight':
    'backbone.stage2.2.linear_conv.conv.weight',
    'backbone.layer9.linear_conv.bn.weight':
    'backbone.stage2.2.linear_conv.bn.weight',
    'backbone.layer9.linear_conv.bn.bias':
    'backbone.stage2.2.linear_conv.bn.bias',
    'backbone.layer9.linear_conv.bn.running_mean':
    'backbone.stage2.2.linear_conv.bn._mean',
    'backbone.layer9.linear_conv.bn.running_var':
    'backbone.stage2.2.linear_conv.bn._variance',
    'backbone.layer10.expand_conv.conv.weight':
    'backbone.stage2.3.expand_conv.conv.weight',
    'backbone.layer10.expand_conv.bn.weight':
    'backbone.stage2.3.expand_conv.bn.weight',
    'backbone.layer10.expand_conv.bn.bias':
    'backbone.stage2.3.expand_conv.bn.bias',
    'backbone.layer10.expand_conv.bn.running_mean':
    'backbone.stage2.3.expand_conv.bn._mean',
    'backbone.layer10.expand_conv.bn.running_var':
    'backbone.stage2.3.expand_conv.bn._variance',
    'backbone.layer10.depthwise_conv.conv.weight':
    'backbone.stage2.3.bottleneck_conv.conv.weight',
    'backbone.layer10.depthwise_conv.bn.weight':
    'backbone.stage2.3.bottleneck_conv.bn.weight',
    'backbone.layer10.depthwise_conv.bn.bias':
    'backbone.stage2.3.bottleneck_conv.bn.bias',
    'backbone.layer10.depthwise_conv.bn.running_mean':
    'backbone.stage2.3.bottleneck_conv.bn._mean',
    'backbone.layer10.depthwise_conv.bn.running_var':
    'backbone.stage2.3.bottleneck_conv.bn._variance',
    'backbone.layer10.linear_conv.conv.weight':
    'backbone.stage2.3.linear_conv.conv.weight',
    'backbone.layer10.linear_conv.bn.weight':
    'backbone.stage2.3.linear_conv.bn.weight',
    'backbone.layer10.linear_conv.bn.bias':
    'backbone.stage2.3.linear_conv.bn.bias',
    'backbone.layer10.linear_conv.bn.running_mean':
    'backbone.stage2.3.linear_conv.bn._mean',
    'backbone.layer10.linear_conv.bn.running_var':
    'backbone.stage2.3.linear_conv.bn._variance',
    'backbone.layer11.expand_conv.conv.weight':
    'backbone.stage2.4.expand_conv.conv.weight',
    'backbone.layer11.expand_conv.bn.weight':
    'backbone.stage2.4.expand_conv.bn.weight',
    'backbone.layer11.expand_conv.bn.bias':
    'backbone.stage2.4.expand_conv.bn.bias',
    'backbone.layer11.expand_conv.bn.running_mean':
    'backbone.stage2.4.expand_conv.bn._mean',
    'backbone.layer11.expand_conv.bn.running_var':
    'backbone.stage2.4.expand_conv.bn._variance',
    'backbone.layer11.depthwise_conv.conv.weight':
    'backbone.stage2.4.bottleneck_conv.conv.weight',
    'backbone.layer11.depthwise_conv.bn.weight':
    'backbone.stage2.4.bottleneck_conv.bn.weight',
    'backbone.layer11.depthwise_conv.bn.bias':
    'backbone.stage2.4.bottleneck_conv.bn.bias',
    'backbone.layer11.depthwise_conv.bn.running_mean':
    'backbone.stage2.4.bottleneck_conv.bn._mean',
    'backbone.layer11.depthwise_conv.bn.running_var':
    'backbone.stage2.4.bottleneck_conv.bn._variance',
    'backbone.layer11.linear_conv.conv.weight':
    'backbone.stage2.4.linear_conv.conv.weight',
    'backbone.layer11.linear_conv.bn.weight':
    'backbone.stage2.4.linear_conv.bn.weight',
    'backbone.layer11.linear_conv.bn.bias':
    'backbone.stage2.4.linear_conv.bn.bias',
    'backbone.layer11.linear_conv.bn.running_mean':
    'backbone.stage2.4.linear_conv.bn._mean',
    'backbone.layer11.linear_conv.bn.running_var':
    'backbone.stage2.4.linear_conv.bn._variance',
    'backbone.layer12.expand_conv.conv.weight':
    'backbone.stage2.5.expand_conv.conv.weight',
    'backbone.layer12.expand_conv.bn.weight':
    'backbone.stage2.5.expand_conv.bn.weight',
    'backbone.layer12.expand_conv.bn.bias':
    'backbone.stage2.5.expand_conv.bn.bias',
    'backbone.layer12.expand_conv.bn.running_mean':
    'backbone.stage2.5.expand_conv.bn._mean',
    'backbone.layer12.expand_conv.bn.running_var':
    'backbone.stage2.5.expand_conv.bn._variance',
    'backbone.layer12.depthwise_conv.conv.weight':
    'backbone.stage2.5.bottleneck_conv.conv.weight',
    'backbone.layer12.depthwise_conv.bn.weight':
    'backbone.stage2.5.bottleneck_conv.bn.weight',
    'backbone.layer12.depthwise_conv.bn.bias':
    'backbone.stage2.5.bottleneck_conv.bn.bias',
    'backbone.layer12.depthwise_conv.bn.running_mean':
    'backbone.stage2.5.bottleneck_conv.bn._mean',
    'backbone.layer12.depthwise_conv.bn.running_var':
    'backbone.stage2.5.bottleneck_conv.bn._variance',
    'backbone.layer12.linear_conv.conv.weight':
    'backbone.stage2.5.linear_conv.conv.weight',
    'backbone.layer12.linear_conv.bn.weight':
    'backbone.stage2.5.linear_conv.bn.weight',
    'backbone.layer12.linear_conv.bn.bias':
    'backbone.stage2.5.linear_conv.bn.bias',
    'backbone.layer12.linear_conv.bn.running_mean':
    'backbone.stage2.5.linear_conv.bn._mean',
    'backbone.layer12.linear_conv.bn.running_var':
    'backbone.stage2.5.linear_conv.bn._variance',
    'backbone.layer13.expand_conv.conv.weight':
    'backbone.stage3.0.expand_conv.conv.weight',
    'backbone.layer13.expand_conv.bn.weight':
    'backbone.stage3.0.expand_conv.bn.weight',
    'backbone.layer13.expand_conv.bn.bias':
    'backbone.stage3.0.expand_conv.bn.bias',
    'backbone.layer13.expand_conv.bn.running_mean':
    'backbone.stage3.0.expand_conv.bn._mean',
    'backbone.layer13.expand_conv.bn.running_var':
    'backbone.stage3.0.expand_conv.bn._variance',
    'backbone.layer13.depthwise_conv.conv.weight':
    'backbone.stage3.0.bottleneck_conv.conv.weight',
    'backbone.layer13.depthwise_conv.bn.weight':
    'backbone.stage3.0.bottleneck_conv.bn.weight',
    'backbone.layer13.depthwise_conv.bn.bias':
    'backbone.stage3.0.bottleneck_conv.bn.bias',
    'backbone.layer13.depthwise_conv.bn.running_mean':
    'backbone.stage3.0.bottleneck_conv.bn._mean',
    'backbone.layer13.depthwise_conv.bn.running_var':
    'backbone.stage3.0.bottleneck_conv.bn._variance',
    'backbone.layer13.linear_conv.conv.weight':
    'backbone.stage3.0.linear_conv.conv.weight',
    'backbone.layer13.linear_conv.bn.weight':
    'backbone.stage3.0.linear_conv.bn.weight',
    'backbone.layer13.linear_conv.bn.bias':
    'backbone.stage3.0.linear_conv.bn.bias',
    'backbone.layer13.linear_conv.bn.running_mean':
    'backbone.stage3.0.linear_conv.bn._mean',
    'backbone.layer13.linear_conv.bn.running_var':
    'backbone.stage3.0.linear_conv.bn._variance',
    'backbone.layer14.expand_conv.conv.weight':
    'backbone.stage3.1.expand_conv.conv.weight',
    'backbone.layer14.expand_conv.bn.weight':
    'backbone.stage3.1.expand_conv.bn.weight',
    'backbone.layer14.expand_conv.bn.bias':
    'backbone.stage3.1.expand_conv.bn.bias',
    'backbone.layer14.expand_conv.bn.running_mean':
    'backbone.stage3.1.expand_conv.bn._mean',
    'backbone.layer14.expand_conv.bn.running_var':
    'backbone.stage3.1.expand_conv.bn._variance',
    'backbone.layer14.depthwise_conv.conv.weight':
    'backbone.stage3.1.bottleneck_conv.conv.weight',
    'backbone.layer14.depthwise_conv.bn.weight':
    'backbone.stage3.1.bottleneck_conv.bn.weight',
    'backbone.layer14.depthwise_conv.bn.bias':
    'backbone.stage3.1.bottleneck_conv.bn.bias',
    'backbone.layer14.depthwise_conv.bn.running_mean':
    'backbone.stage3.1.bottleneck_conv.bn._mean',
    'backbone.layer14.depthwise_conv.bn.running_var':
    'backbone.stage3.1.bottleneck_conv.bn._variance',
    'backbone.layer14.linear_conv.conv.weight':
    'backbone.stage3.1.linear_conv.conv.weight',
    'backbone.layer14.linear_conv.bn.weight':
    'backbone.stage3.1.linear_conv.bn.weight',
    'backbone.layer14.linear_conv.bn.bias':
    'backbone.stage3.1.linear_conv.bn.bias',
    'backbone.layer14.linear_conv.bn.running_mean':
    'backbone.stage3.1.linear_conv.bn._mean',
    'backbone.layer14.linear_conv.bn.running_var':
    'backbone.stage3.1.linear_conv.bn._variance',
    'backbone.layer15.expand_conv.conv.weight':
    'backbone.stage3.2.expand_conv.conv.weight',
    'backbone.layer15.expand_conv.bn.weight':
    'backbone.stage3.2.expand_conv.bn.weight',
    'backbone.layer15.expand_conv.bn.bias':
    'backbone.stage3.2.expand_conv.bn.bias',
    'backbone.layer15.expand_conv.bn.running_mean':
    'backbone.stage3.2.expand_conv.bn._mean',
    'backbone.layer15.expand_conv.bn.running_var':
    'backbone.stage3.2.expand_conv.bn._variance',
    'backbone.layer15.depthwise_conv.conv.weight':
    'backbone.stage3.2.bottleneck_conv.conv.weight',
    'backbone.layer15.depthwise_conv.bn.weight':
    'backbone.stage3.2.bottleneck_conv.bn.weight',
    'backbone.layer15.depthwise_conv.bn.bias':
    'backbone.stage3.2.bottleneck_conv.bn.bias',
    'backbone.layer15.depthwise_conv.bn.running_mean':
    'backbone.stage3.2.bottleneck_conv.bn._mean',
    'backbone.layer15.depthwise_conv.bn.running_var':
    'backbone.stage3.2.bottleneck_conv.bn._variance',
    'backbone.layer15.linear_conv.conv.weight':
    'backbone.stage3.2.linear_conv.conv.weight',
    'backbone.layer15.linear_conv.bn.weight':
    'backbone.stage3.2.linear_conv.bn.weight',
    'backbone.layer15.linear_conv.bn.bias':
    'backbone.stage3.2.linear_conv.bn.bias',
    'backbone.layer15.linear_conv.bn.running_mean':
    'backbone.stage3.2.linear_conv.bn._mean',
    'backbone.layer15.linear_conv.bn.running_var':
    'backbone.stage3.2.linear_conv.bn._variance',
    'backbone.layer16.conv.weight': 'backbone.stage3.3.conv.weight',
    'backbone.layer16.bn.weight': 'backbone.stage3.3.bn.weight',
    'backbone.layer16.bn.bias': 'backbone.stage3.3.bn.bias',
    'backbone.layer16.bn.running_mean': 'backbone.stage3.3.bn._mean',
    'backbone.layer16.bn.running_var': 'backbone.stage3.3.bn._variance',
    'neck.ins_conv.0.in_conv.conv.weight': 'neck.ins_conv.0.in_conv.weight',
    'neck.ins_conv.0.se.conv1.conv.weight':
    'neck.ins_conv.0.se_block.conv1.weight',
    'neck.ins_conv.0.se.conv1.conv.bias':
    'neck.ins_conv.0.se_block.conv1.bias',
    'neck.ins_conv.0.se.conv2.conv.weight':
    'neck.ins_conv.0.se_block.conv2.weight',
    'neck.ins_conv.0.se.conv2.conv.bias':
    'neck.ins_conv.0.se_block.conv2.bias',
    'neck.ins_conv.1.in_conv.conv.weight': 'neck.ins_conv.1.in_conv.weight',
    'neck.ins_conv.1.se.conv1.conv.weight':
    'neck.ins_conv.1.se_block.conv1.weight',
    'neck.ins_conv.1.se.conv1.conv.bias':
    'neck.ins_conv.1.se_block.conv1.bias',
    'neck.ins_conv.1.se.conv2.conv.weight':
    'neck.ins_conv.1.se_block.conv2.weight',
    'neck.ins_conv.1.se.conv2.conv.bias': 'neck.ins_conv.1.se_block.conv2.'
    'bias',
    'neck.ins_conv.2.in_conv.conv.weight': 'neck.ins_conv.2.in_conv.weight',
    'neck.ins_conv.2.se.conv1.conv.weight':
    'neck.ins_conv.2.se_block.conv1.weight',
    'neck.ins_conv.2.se.conv1.conv.bias': 'neck.ins_conv.2.se_block.conv1.'
    'bias',
    'neck.ins_conv.2.se.conv2.conv.weight':
    'neck.ins_conv.2.se_block.conv2.weight',
    'neck.ins_conv.2.se.conv2.conv.bias': 'neck.ins_conv.2.se_block.conv2.'
    'bias',
    'neck.ins_conv.3.in_conv.conv.weight': 'neck.ins_conv.3.in_conv.weight',
    'neck.ins_conv.3.se.conv1.conv.weight':
    'neck.ins_conv.3.se_block.conv1.weight',
    'neck.ins_conv.3.se.conv1.conv.bias': 'neck.ins_conv.3.se_block.conv1.'
    'bias',
    'neck.ins_conv.3.se.conv2.conv.weight':
    'neck.ins_conv.3.se_block.conv2.weight',
    'neck.ins_conv.3.se.conv2.conv.bias': 'neck.ins_conv.3.se_block.conv2.'
    'bias',
    'neck.inp_conv.0.in_conv.conv.weight': 'neck.inp_conv.0.in_conv.weight',
    'neck.inp_conv.0.se.conv1.conv.weight':
    'neck.inp_conv.0.se_block.conv1.weight',
    'neck.inp_conv.0.se.conv1.conv.bias': 'neck.inp_conv.0.se_block.conv1.'
    'bias',
    'neck.inp_conv.0.se.conv2.conv.weight':
    'neck.inp_conv.0.se_block.conv2.weight',
    'neck.inp_conv.0.se.conv2.conv.bias': 'neck.inp_conv.0.se_block.conv2.'
    'bias',
    'neck.inp_conv.1.in_conv.conv.weight': 'neck.inp_conv.1.in_conv.weight',
    'neck.inp_conv.1.se.conv1.conv.weight':
    'neck.inp_conv.1.se_block.conv1.weight',
    'neck.inp_conv.1.se.conv1.conv.bias': 'neck.inp_conv.1.se_block.conv1.'
    'bias',
    'neck.inp_conv.1.se.conv2.conv.weight':
    'neck.inp_conv.1.se_block.conv2.weight',
    'neck.inp_conv.1.se.conv2.conv.bias': 'neck.inp_conv.1.se_block.conv2.'
    'bias',
    'neck.inp_conv.2.in_conv.conv.weight': 'neck.inp_conv.2.in_conv.weight',
    'neck.inp_conv.2.se.conv1.conv.weight':
    'neck.inp_conv.2.se_block.conv1.weight',
    'neck.inp_conv.2.se.conv1.conv.bias': 'neck.inp_conv.2.se_block.conv1.'
    'bias',
    'neck.inp_conv.2.se.conv2.conv.weight':
    'neck.inp_conv.2.se_block.conv2.weight',
    'neck.inp_conv.2.se.conv2.conv.bias': 'neck.inp_conv.2.se_block.conv2.'
    'bias',
    'neck.inp_conv.3.in_conv.conv.weight': 'neck.inp_conv.3.in_conv.weight',
    'neck.inp_conv.3.se.conv1.conv.weight':
    'neck.inp_conv.3.se_block.conv1.weight',
    'neck.inp_conv.3.se.conv1.conv.bias': 'neck.inp_conv.3.se_block.conv1.'
    'bias',
    'neck.inp_conv.3.se.conv2.conv.weight':
    'neck.inp_conv.3.se_block.conv2.weight',
    'neck.inp_conv.3.se.conv2.conv.bias': 'neck.inp_conv.3.se_block.conv2.'
    'bias',
    'det_head.binarize.0.weight': 'head.binarize.conv1.weight',
    'det_head.binarize.1.weight': 'head.binarize.conv_bn1.weight',
    'det_head.binarize.1.bias': 'head.binarize.conv_bn1.bias',
    'det_head.binarize.1.running_mean': 'head.binarize.conv_bn1._mean',
    'det_head.binarize.1.running_var': 'head.binarize.conv_bn1._variance',
    'det_head.binarize.3.weight': 'head.binarize.conv2.weight',
    'det_head.binarize.3.bias': 'head.binarize.conv2.bias',
    'det_head.binarize.4.weight': 'head.binarize.conv_bn2.weight',
    'det_head.binarize.4.bias': 'head.binarize.conv_bn2.bias',
    'det_head.binarize.4.running_mean': 'head.binarize.conv_bn2._mean',
    'det_head.binarize.4.running_var': 'head.binarize.conv_bn2._variance',
    'det_head.binarize.6.weight': 'head.binarize.conv3.weight',
    'det_head.binarize.6.bias': 'head.binarize.conv3.bias',
    'det_head.threshold.0.weight': 'head.thresh.conv1.weight',
    'det_head.threshold.1.weight': 'head.thresh.conv_bn1.weight',
    'det_head.threshold.1.bias': 'head.thresh.conv_bn1.bias',
    'det_head.threshold.1.running_mean': 'head.thresh.conv_bn1._mean',
    'det_head.threshold.1.running_var': 'head.thresh.conv_bn1._variance',
    'det_head.threshold.3.weight': 'head.thresh.conv2.weight',
    'det_head.threshold.3.bias': 'head.thresh.conv2.bias',
    'det_head.threshold.4.weight': 'head.thresh.conv_bn2.weight',
    'det_head.threshold.4.bias': 'head.thresh.conv_bn2.bias',
    'det_head.threshold.4.running_mean': 'head.thresh.conv_bn2._mean',
    'det_head.threshold.4.running_var': 'head.thresh.conv_bn2._variance',
    'det_head.threshold.6.weight': 'head.thresh.conv3.weight',
    'det_head.threshold.6.bias': 'head.thresh.conv3.bias'
}


def map_state(ppocr_state: dict) -> dict:
    torch_state = {}
    print(ppocr_state)
    for key in STATE_MAP:
        torch_state[key] = torch.from_numpy(ppocr_state[STATE_MAP[key]])
    return torch_state


def main():
    register_all_modules(init_default_scope=False)
    torch_model_path = '../../pretrained/ppocr_v3_db.pth'
    ppocr_model_path = '../../pretrained/ch_PP-OCRv3_det_distill_train/student'
    cfg = Config.fromfile('../../configs/_base_/models/dbnet_little.py')
    # model = build_detector(cfg.model)
    model = MODELS.build(cfg.model)
    ppocr_state = fluid.io.load_program_state(ppocr_model_path)
    torch_state = map_state(ppocr_state)
    torch.save(torch_state, torch_model_path)

    model.load_state_dict(torch_state)
    img = mmcv.imread('../../demo/2.jpg')
    cfg = Config.fromfile(
        '../../configs/_base_/datasets/det_pipeline.py').test_pipeline_1333_736
    cfg.pop(0)
    test_pipeline = Compose(cfg)
    inputs = test_pipeline({
        'img': img,
        'filename': 'test.jpg',
        'ori_filename': 'test.jpg',
        'ori_shape': img.shape
    })
    inputs['inputs'] = [inputs['inputs']]
    inputs['data_samples'] = [inputs['data_samples']]
    model.eval()
    with torch.no_grad():
        torch_outputs = model.det_head(
            model.extract_feat(
                model.data_preprocessor(inputs, True)['inputs']))

    model_file_path = '../../pretrained/ch_PP-OCRv3_det_infer/' \
                      'inference.pdmodel'
    params_file_path = '../../pretrained/ch_PP-OCRv3_det_infer/' \
                       'inference.pdiparams'
    config = inference.Config(model_file_path, params_file_path)
    predictor = inference.create_predictor(config)

    input_names = predictor.get_input_names()
    output_names = predictor.get_output_names()
    output_tensors = []
    for name in input_names:
        input_tensor = predictor.get_input_handle(name)
    for output_name in output_names:
        output_tensor = predictor.get_output_handle(output_name)
        output_tensors.append(output_tensor)
    input_tensor.copy_from_cpu(
        model.data_preprocessor(inputs, True)['inputs'].numpy())
    predictor.run()
    ppocr_outputs = []
    for output_tensor in output_tensors:
        output = output_tensor.copy_to_cpu()
        ppocr_outputs.append(output)
    print(np.abs(ppocr_outputs[0] - torch_outputs.numpy()).mean())


if __name__ == '__main__':
    main()
