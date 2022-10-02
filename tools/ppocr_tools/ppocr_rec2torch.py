# @Time : 2022/7/31 13:43
# @Author : songweinan
# @Software: PyCharm
# 自能成羽翼，何必仰云梯。
# ======================================================================================================================
import mmcv
from mmengine import Config
import numpy as np
import paddle.fluid as fluid
import torch
from mmengine.dataset.base_dataset import Compose
from mmengine.registry import MODELS
from thops.utils import register_all_modules
from paddle import inference


def map_state(torch_state: dict, ppocr_state: dict) -> dict:
    transpose = [
        'Student.head.ctc_encoder.encoder.svtr_block.0.mixer.qkv.weight',
        'Student.head.ctc_encoder.encoder.svtr_block.0.mixer.proj.weight',
        'Student.head.ctc_encoder.encoder.svtr_block.0.mlp.fc1.weight',
        'Student.head.ctc_encoder.encoder.svtr_block.0.mlp.fc2.weight',
        'Student.head.ctc_encoder.encoder.svtr_block.1.mixer.qkv.weight',
        'Student.head.ctc_encoder.encoder.svtr_block.1.mixer.proj.weight',
        'Student.head.ctc_encoder.encoder.svtr_block.1.mlp.fc1.weight',
        'Student.head.ctc_encoder.encoder.svtr_block.1.mlp.fc2.weight',
    ]
    state_map = {
        'backbone.conv1.conv.weight':
        'Student.backbone.conv1._conv.weight',
        'backbone.conv1.bn.weight':
        'Student.backbone.conv1._batch_norm.weight',
        'backbone.conv1.bn.bias':
        'Student.backbone.conv1._batch_norm.bias',
        'backbone.conv1.bn.running_mean':
        'Student.backbone.conv1._batch_norm._mean',
        'backbone.conv1.bn.running_var':
        'Student.backbone.conv1._batch_norm._variance',
        'backbone.block_list.0.depthwise_conv.conv.weight':
        'Student.backbone.block_list.0._depthwise_conv._conv.weight',
        'backbone.block_list.0.depthwise_conv.bn.weight':
        'Student.backbone.block_list.0._depthwise_conv._batch_norm.weight',
        'backbone.block_list.0.depthwise_conv.bn.bias':
        'Student.backbone.block_list.0._depthwise_conv._batch_norm.bias',
        'backbone.block_list.0.depthwise_conv.bn.running_mean':
        'Student.backbone.block_list.0._depthwise_conv._batch_norm._mean',
        'backbone.block_list.0.depthwise_conv.bn.running_var':
        'Student.backbone.block_list.0._depthwise_conv._batch_norm._variance',
        'backbone.block_list.0.linear_conv.conv.weight':
        'Student.backbone.block_list.0._pointwise_conv._conv.weight',
        'backbone.block_list.0.linear_conv.bn.weight':
        'Student.backbone.block_list.0._pointwise_conv._batch_norm.weight',
        'backbone.block_list.0.linear_conv.bn.bias':
        'Student.backbone.block_list.0._pointwise_conv._batch_norm.bias',
        'backbone.block_list.0.linear_conv.bn.running_mean':
        'Student.backbone.block_list.0._pointwise_conv._batch_norm._mean',
        'backbone.block_list.0.linear_conv.bn.running_var':
        'Student.backbone.block_list.0._pointwise_conv._batch_norm._variance',
        'backbone.block_list.1.depthwise_conv.conv.weight':
        'Student.backbone.block_list.1._depthwise_conv._conv.weight',
        'backbone.block_list.1.depthwise_conv.bn.weight':
        'Student.backbone.block_list.1._depthwise_conv._batch_norm.weight',
        'backbone.block_list.1.depthwise_conv.bn.bias':
        'Student.backbone.block_list.1._depthwise_conv._batch_norm.bias',
        'backbone.block_list.1.depthwise_conv.bn.running_mean':
        'Student.backbone.block_list.1._depthwise_conv._batch_norm._mean',
        'backbone.block_list.1.depthwise_conv.bn.running_var':
        'Student.backbone.block_list.1._depthwise_conv._batch_norm._variance',
        'backbone.block_list.1.linear_conv.conv.weight':
        'Student.backbone.block_list.1._pointwise_conv._conv.weight',
        'backbone.block_list.1.linear_conv.bn.weight':
        'Student.backbone.block_list.1._pointwise_conv._batch_norm.weight',
        'backbone.block_list.1.linear_conv.bn.bias':
        'Student.backbone.block_list.1._pointwise_conv._batch_norm.bias',
        'backbone.block_list.1.linear_conv.bn.running_mean':
        'Student.backbone.block_list.1._pointwise_conv._batch_norm._mean',
        'backbone.block_list.1.linear_conv.bn.running_var':
        'Student.backbone.block_list.1._pointwise_conv._batch_norm._variance',
        'backbone.block_list.2.depthwise_conv.conv.weight':
        'Student.backbone.block_list.2._depthwise_conv._conv.weight',
        'backbone.block_list.2.depthwise_conv.bn.weight':
        'Student.backbone.block_list.2._depthwise_conv._batch_norm.weight',
        'backbone.block_list.2.depthwise_conv.bn.bias':
        'Student.backbone.block_list.2._depthwise_conv._batch_norm.bias',
        'backbone.block_list.2.depthwise_conv.bn.running_mean':
        'Student.backbone.block_list.2._depthwise_conv._batch_norm._mean',
        'backbone.block_list.2.depthwise_conv.bn.running_var':
        'Student.backbone.block_list.2._depthwise_conv._batch_norm._variance',
        'backbone.block_list.2.linear_conv.conv.weight':
        'Student.backbone.block_list.2._pointwise_conv._conv.weight',
        'backbone.block_list.2.linear_conv.bn.weight':
        'Student.backbone.block_list.2._pointwise_conv._batch_norm.weight',
        'backbone.block_list.2.linear_conv.bn.bias':
        'Student.backbone.block_list.2._pointwise_conv._batch_norm.bias',
        'backbone.block_list.2.linear_conv.bn.running_mean':
        'Student.backbone.block_list.2._pointwise_conv._batch_norm._mean',
        'backbone.block_list.2.linear_conv.bn.running_var':
        'Student.backbone.block_list.2._pointwise_conv._batch_norm._variance',
        'backbone.block_list.3.depthwise_conv.conv.weight':
        'Student.backbone.block_list.3._depthwise_conv._conv.weight',
        'backbone.block_list.3.depthwise_conv.bn.weight':
        'Student.backbone.block_list.3._depthwise_conv._batch_norm.weight',
        'backbone.block_list.3.depthwise_conv.bn.bias':
        'Student.backbone.block_list.3._depthwise_conv._batch_norm.bias',
        'backbone.block_list.3.depthwise_conv.bn.running_mean':
        'Student.backbone.block_list.3._depthwise_conv._batch_norm._mean',
        'backbone.block_list.3.depthwise_conv.bn.running_var':
        'Student.backbone.block_list.3._depthwise_conv._batch_norm._variance',
        'backbone.block_list.3.linear_conv.conv.weight':
        'Student.backbone.block_list.3._pointwise_conv._conv.weight',
        'backbone.block_list.3.linear_conv.bn.weight':
        'Student.backbone.block_list.3._pointwise_conv._batch_norm.weight',
        'backbone.block_list.3.linear_conv.bn.bias':
        'Student.backbone.block_list.3._pointwise_conv._batch_norm.bias',
        'backbone.block_list.3.linear_conv.bn.running_mean':
        'Student.backbone.block_list.3._pointwise_conv._batch_norm._mean',
        'backbone.block_list.3.linear_conv.bn.running_var':
        'Student.backbone.block_list.3._pointwise_conv._batch_norm._variance',
        'backbone.block_list.4.depthwise_conv.conv.weight':
        'Student.backbone.block_list.4._depthwise_conv._conv.weight',
        'backbone.block_list.4.depthwise_conv.bn.weight':
        'Student.backbone.block_list.4._depthwise_conv._batch_norm.weight',
        'backbone.block_list.4.depthwise_conv.bn.bias':
        'Student.backbone.block_list.4._depthwise_conv._batch_norm.bias',
        'backbone.block_list.4.depthwise_conv.bn.running_mean':
        'Student.backbone.block_list.4._depthwise_conv._batch_norm._mean',
        'backbone.block_list.4.depthwise_conv.bn.running_var':
        'Student.backbone.block_list.4._depthwise_conv._batch_norm._variance',
        'backbone.block_list.4.linear_conv.conv.weight':
        'Student.backbone.block_list.4._pointwise_conv._conv.weight',
        'backbone.block_list.4.linear_conv.bn.weight':
        'Student.backbone.block_list.4._pointwise_conv._batch_norm.weight',
        'backbone.block_list.4.linear_conv.bn.bias':
        'Student.backbone.block_list.4._pointwise_conv._batch_norm.bias',
        'backbone.block_list.4.linear_conv.bn.running_mean':
        'Student.backbone.block_list.4._pointwise_conv._batch_norm._mean',
        'backbone.block_list.4.linear_conv.bn.running_var':
        'Student.backbone.block_list.4._pointwise_conv._batch_norm._variance',
        'backbone.block_list.5.depthwise_conv.conv.weight':
        'Student.backbone.block_list.5._depthwise_conv._conv.weight',
        'backbone.block_list.5.depthwise_conv.bn.weight':
        'Student.backbone.block_list.5._depthwise_conv._batch_norm.weight',
        'backbone.block_list.5.depthwise_conv.bn.bias':
        'Student.backbone.block_list.5._depthwise_conv._batch_norm.bias',
        'backbone.block_list.5.depthwise_conv.bn.running_mean':
        'Student.backbone.block_list.5._depthwise_conv._batch_norm._mean',
        'backbone.block_list.5.depthwise_conv.bn.running_var':
        'Student.backbone.block_list.5._depthwise_conv._batch_norm._variance',
        'backbone.block_list.5.linear_conv.conv.weight':
        'Student.backbone.block_list.5._pointwise_conv._conv.weight',
        'backbone.block_list.5.linear_conv.bn.weight':
        'Student.backbone.block_list.5._pointwise_conv._batch_norm.weight',
        'backbone.block_list.5.linear_conv.bn.bias':
        'Student.backbone.block_list.5._pointwise_conv._batch_norm.bias',
        'backbone.block_list.5.linear_conv.bn.running_mean':
        'Student.backbone.block_list.5._pointwise_conv._batch_norm._mean',
        'backbone.block_list.5.linear_conv.bn.running_var':
        'Student.backbone.block_list.5._pointwise_conv._batch_norm._variance',
        'backbone.block_list.6.depthwise_conv.conv.weight':
        'Student.backbone.block_list.6._depthwise_conv._conv.weight',
        'backbone.block_list.6.depthwise_conv.bn.weight':
        'Student.backbone.block_list.6._depthwise_conv._batch_norm.weight',
        'backbone.block_list.6.depthwise_conv.bn.bias':
        'Student.backbone.block_list.6._depthwise_conv._batch_norm.bias',
        'backbone.block_list.6.depthwise_conv.bn.running_mean':
        'Student.backbone.block_list.6._depthwise_conv._batch_norm._mean',
        'backbone.block_list.6.depthwise_conv.bn.running_var':
        'Student.backbone.block_list.6._depthwise_conv._batch_norm._variance',
        'backbone.block_list.6.linear_conv.conv.weight':
        'Student.backbone.block_list.6._pointwise_conv._conv.weight',
        'backbone.block_list.6.linear_conv.bn.weight':
        'Student.backbone.block_list.6._pointwise_conv._batch_norm.weight',
        'backbone.block_list.6.linear_conv.bn.bias':
        'Student.backbone.block_list.6._pointwise_conv._batch_norm.bias',
        'backbone.block_list.6.linear_conv.bn.running_mean':
        'Student.backbone.block_list.6._pointwise_conv._batch_norm._mean',
        'backbone.block_list.6.linear_conv.bn.running_var':
        'Student.backbone.block_list.6._pointwise_conv._batch_norm._variance',
        'backbone.block_list.7.depthwise_conv.conv.weight':
        'Student.backbone.block_list.7._depthwise_conv._conv.weight',
        'backbone.block_list.7.depthwise_conv.bn.weight':
        'Student.backbone.block_list.7._depthwise_conv._batch_norm.weight',
        'backbone.block_list.7.depthwise_conv.bn.bias':
        'Student.backbone.block_list.7._depthwise_conv._batch_norm.bias',
        'backbone.block_list.7.depthwise_conv.bn.running_mean':
        'Student.backbone.block_list.7._depthwise_conv._batch_norm._mean',
        'backbone.block_list.7.depthwise_conv.bn.running_var':
        'Student.backbone.block_list.7._depthwise_conv._batch_norm._variance',
        'backbone.block_list.7.linear_conv.conv.weight':
        'Student.backbone.block_list.7._pointwise_conv._conv.weight',
        'backbone.block_list.7.linear_conv.bn.weight':
        'Student.backbone.block_list.7._pointwise_conv._batch_norm.weight',
        'backbone.block_list.7.linear_conv.bn.bias':
        'Student.backbone.block_list.7._pointwise_conv._batch_norm.bias',
        'backbone.block_list.7.linear_conv.bn.running_mean':
        'Student.backbone.block_list.7._pointwise_conv._batch_norm._mean',
        'backbone.block_list.7.linear_conv.bn.running_var':
        'Student.backbone.block_list.7._pointwise_conv._batch_norm._variance',
        'backbone.block_list.8.depthwise_conv.conv.weight':
        'Student.backbone.block_list.8._depthwise_conv._conv.weight',
        'backbone.block_list.8.depthwise_conv.bn.weight':
        'Student.backbone.block_list.8._depthwise_conv._batch_norm.weight',
        'backbone.block_list.8.depthwise_conv.bn.bias':
        'Student.backbone.block_list.8._depthwise_conv._batch_norm.bias',
        'backbone.block_list.8.depthwise_conv.bn.running_mean':
        'Student.backbone.block_list.8._depthwise_conv._batch_norm._mean',
        'backbone.block_list.8.depthwise_conv.bn.running_var':
        'Student.backbone.block_list.8._depthwise_conv._batch_norm._variance',
        'backbone.block_list.8.linear_conv.conv.weight':
        'Student.backbone.block_list.8._pointwise_conv._conv.weight',
        'backbone.block_list.8.linear_conv.bn.weight':
        'Student.backbone.block_list.8._pointwise_conv._batch_norm.weight',
        'backbone.block_list.8.linear_conv.bn.bias':
        'Student.backbone.block_list.8._pointwise_conv._batch_norm.bias',
        'backbone.block_list.8.linear_conv.bn.running_mean':
        'Student.backbone.block_list.8._pointwise_conv._batch_norm._mean',
        'backbone.block_list.8.linear_conv.bn.running_var':
        'Student.backbone.block_list.8._pointwise_conv._batch_norm._variance',
        'backbone.block_list.9.depthwise_conv.conv.weight':
        'Student.backbone.block_list.9._depthwise_conv._conv.weight',
        'backbone.block_list.9.depthwise_conv.bn.weight':
        'Student.backbone.block_list.9._depthwise_conv._batch_norm.weight',
        'backbone.block_list.9.depthwise_conv.bn.bias':
        'Student.backbone.block_list.9._depthwise_conv._batch_norm.bias',
        'backbone.block_list.9.depthwise_conv.bn.running_mean':
        'Student.backbone.block_list.9._depthwise_conv._batch_norm._mean',
        'backbone.block_list.9.depthwise_conv.bn.running_var':
        'Student.backbone.block_list.9._depthwise_conv._batch_norm._variance',
        'backbone.block_list.9.linear_conv.conv.weight':
        'Student.backbone.block_list.9._pointwise_conv._conv.weight',
        'backbone.block_list.9.linear_conv.bn.weight':
        'Student.backbone.block_list.9._pointwise_conv._batch_norm.weight',
        'backbone.block_list.9.linear_conv.bn.bias':
        'Student.backbone.block_list.9._pointwise_conv._batch_norm.bias',
        'backbone.block_list.9.linear_conv.bn.running_mean':
        'Student.backbone.block_list.9._pointwise_conv._batch_norm._mean',
        'backbone.block_list.9.linear_conv.bn.running_var':
        'Student.backbone.block_list.9._pointwise_conv._batch_norm._variance',
        'backbone.block_list.10.depthwise_conv.conv.weight':
        'Student.backbone.block_list.10._depthwise_conv._conv.weight',
        'backbone.block_list.10.depthwise_conv.bn.weight':
        'Student.backbone.block_list.10._depthwise_conv._batch_norm.weight',
        'backbone.block_list.10.depthwise_conv.bn.bias':
        'Student.backbone.block_list.10._depthwise_conv._batch_norm.bias',
        'backbone.block_list.10.depthwise_conv.bn.running_mean':
        'Student.backbone.block_list.10._depthwise_conv._batch_norm._mean',
        'backbone.block_list.10.depthwise_conv.bn.running_var':
        'Student.backbone.block_list.10._depthwise_conv._batch_norm._variance',
        'backbone.block_list.10.linear_conv.conv.weight':
        'Student.backbone.block_list.10._pointwise_conv._conv.weight',
        'backbone.block_list.10.linear_conv.bn.weight':
        'Student.backbone.block_list.10._pointwise_conv._batch_norm.weight',
        'backbone.block_list.10.linear_conv.bn.bias':
        'Student.backbone.block_list.10._pointwise_conv._batch_norm.bias',
        'backbone.block_list.10.linear_conv.bn.running_mean':
        'Student.backbone.block_list.10._pointwise_conv._batch_norm._mean',
        'backbone.block_list.10.linear_conv.bn.running_var':
        'Student.backbone.block_list.10._pointwise_conv._batch_norm._variance',
        'backbone.block_list.11.depthwise_conv.conv.weight':
        'Student.backbone.block_list.11._depthwise_conv._conv.weight',
        'backbone.block_list.11.depthwise_conv.bn.weight':
        'Student.backbone.block_list.11._depthwise_conv._batch_norm.weight',
        'backbone.block_list.11.depthwise_conv.bn.bias':
        'Student.backbone.block_list.11._depthwise_conv._batch_norm.bias',
        'backbone.block_list.11.depthwise_conv.bn.running_mean':
        'Student.backbone.block_list.11._depthwise_conv._batch_norm._mean',
        'backbone.block_list.11.depthwise_conv.bn.running_var':
        'Student.backbone.block_list.11._depthwise_conv._batch_norm._variance',
        'backbone.block_list.11.se.conv1.conv.weight':
        'Student.backbone.block_list.11._se.conv1.weight',
        'backbone.block_list.11.se.conv1.conv.bias':
        'Student.backbone.block_list.11._se.conv1.bias',
        'backbone.block_list.11.se.conv2.conv.weight':
        'Student.backbone.block_list.11._se.conv2.weight',
        'backbone.block_list.11.se.conv2.conv.bias':
        'Student.backbone.block_list.11._se.conv2.bias',
        'backbone.block_list.11.linear_conv.conv.weight':
        'Student.backbone.block_list.11._pointwise_conv._conv.weight',
        'backbone.block_list.11.linear_conv.bn.weight':
        'Student.backbone.block_list.11._pointwise_conv._batch_norm.weight',
        'backbone.block_list.11.linear_conv.bn.bias':
        'Student.backbone.block_list.11._pointwise_conv._batch_norm.bias',
        'backbone.block_list.11.linear_conv.bn.running_mean':
        'Student.backbone.block_list.11._pointwise_conv._batch_norm._mean',
        'backbone.block_list.11.linear_conv.bn.running_var':
        'Student.backbone.block_list.11._pointwise_conv._batch_norm._variance',
        'backbone.block_list.12.depthwise_conv.conv.weight':
        'Student.backbone.block_list.12._depthwise_conv._conv.weight',
        'backbone.block_list.12.depthwise_conv.bn.weight':
        'Student.backbone.block_list.12._depthwise_conv._batch_norm.weight',
        'backbone.block_list.12.depthwise_conv.bn.bias':
        'Student.backbone.block_list.12._depthwise_conv._batch_norm.bias',
        'backbone.block_list.12.depthwise_conv.bn.running_mean':
        'Student.backbone.block_list.12._depthwise_conv._batch_norm._mean',
        'backbone.block_list.12.depthwise_conv.bn.running_var':
        'Student.backbone.block_list.12._depthwise_conv._batch_norm._variance',
        'backbone.block_list.12.se.conv1.conv.weight':
        'Student.backbone.block_list.12._se.conv1.weight',
        'backbone.block_list.12.se.conv1.conv.bias':
        'Student.backbone.block_list.12._se.conv1.bias',
        'backbone.block_list.12.se.conv2.conv.weight':
        'Student.backbone.block_list.12._se.conv2.weight',
        'backbone.block_list.12.se.conv2.conv.bias':
        'Student.backbone.block_list.12._se.conv2.bias',
        'backbone.block_list.12.linear_conv.conv.weight':
        'Student.backbone.block_list.12._pointwise_conv._conv.weight',
        'backbone.block_list.12.linear_conv.bn.weight':
        'Student.backbone.block_list.12._pointwise_conv._batch_norm.weight',
        'backbone.block_list.12.linear_conv.bn.bias':
        'Student.backbone.block_list.12._pointwise_conv._batch_norm.bias',
        'backbone.block_list.12.linear_conv.bn.running_mean':
        'Student.backbone.block_list.12._pointwise_conv._batch_norm._mean',
        'backbone.block_list.12.linear_conv.bn.running_var':
        'Student.backbone.block_list.12._pointwise_conv._batch_norm._variance',
        'encoder.conv1.conv.weight':
        'Student.head.ctc_encoder.encoder.conv1.conv.weight',
        'encoder.conv1.bn.weight':
        'Student.head.ctc_encoder.encoder.conv1.norm.weight',
        'encoder.conv1.bn.bias':
        'Student.head.ctc_encoder.encoder.conv1.norm.bias',
        'encoder.conv1.bn.running_mean':
        'Student.head.ctc_encoder.encoder.conv1.norm._mean',
        'encoder.conv1.bn.running_var':
        'Student.head.ctc_encoder.encoder.conv1.norm._variance',
        'encoder.conv2.conv.weight':
        'Student.head.ctc_encoder.encoder.conv2.conv.weight',
        'encoder.conv2.bn.weight':
        'Student.head.ctc_encoder.encoder.conv2.norm.weight',
        'encoder.conv2.bn.bias':
        'Student.head.ctc_encoder.encoder.conv2.norm.bias',
        'encoder.conv2.bn.running_mean':
        'Student.head.ctc_encoder.encoder.conv2.norm._mean',
        'encoder.conv2.bn.running_var':
        'Student.head.ctc_encoder.encoder.conv2.norm._variance',
        'encoder.svtr_block.0.norm1.weight':
        'Student.head.ctc_encoder.encoder.svtr_block.0.norm1.weight',
        'encoder.svtr_block.0.norm1.bias':
        'Student.head.ctc_encoder.encoder.svtr_block.0.norm1.bias',
        'encoder.svtr_block.0.mixer.qkv.weight':
        'Student.head.ctc_encoder.encoder.svtr_block.0.mixer.qkv.weight',
        'encoder.svtr_block.0.mixer.qkv.bias':
        'Student.head.ctc_encoder.encoder.svtr_block.0.mixer.qkv.bias',
        'encoder.svtr_block.0.mixer.proj.weight':
        'Student.head.ctc_encoder.encoder.svtr_block.0.mixer.proj.weight',
        'encoder.svtr_block.0.mixer.proj.bias':
        'Student.head.ctc_encoder.encoder.svtr_block.0.mixer.proj.bias',
        'encoder.svtr_block.0.norm2.weight':
        'Student.head.ctc_encoder.encoder.svtr_block.0.norm2.weight',
        'encoder.svtr_block.0.norm2.bias':
        'Student.head.ctc_encoder.encoder.svtr_block.0.norm2.bias',
        'encoder.svtr_block.0.mlp.fc1.weight':
        'Student.head.ctc_encoder.encoder.svtr_block.0.mlp.fc1.weight',
        'encoder.svtr_block.0.mlp.fc1.bias':
        'Student.head.ctc_encoder.encoder.svtr_block.0.mlp.fc1.bias',
        'encoder.svtr_block.0.mlp.fc2.weight':
        'Student.head.ctc_encoder.encoder.svtr_block.0.mlp.fc2.weight',
        'encoder.svtr_block.0.mlp.fc2.bias':
        'Student.head.ctc_encoder.encoder.svtr_block.0.mlp.fc2.bias',
        'encoder.svtr_block.1.norm1.weight':
        'Student.head.ctc_encoder.encoder.svtr_block.1.norm1.weight',
        'encoder.svtr_block.1.norm1.bias':
        'Student.head.ctc_encoder.encoder.svtr_block.1.norm1.bias',
        'encoder.svtr_block.1.mixer.qkv.weight':
        'Student.head.ctc_encoder.encoder.svtr_block.1.mixer.qkv.weight',
        'encoder.svtr_block.1.mixer.qkv.bias':
        'Student.head.ctc_encoder.encoder.svtr_block.1.mixer.qkv.bias',
        'encoder.svtr_block.1.mixer.proj.weight':
        'Student.head.ctc_encoder.encoder.svtr_block.1.mixer.proj.weight',
        'encoder.svtr_block.1.mixer.proj.bias':
        'Student.head.ctc_encoder.encoder.svtr_block.1.mixer.proj.bias',
        'encoder.svtr_block.1.norm2.weight':
        'Student.head.ctc_encoder.encoder.svtr_block.1.norm2.weight',
        'encoder.svtr_block.1.norm2.bias':
        'Student.head.ctc_encoder.encoder.svtr_block.1.norm2.bias',
        'encoder.svtr_block.1.mlp.fc1.weight':
        'Student.head.ctc_encoder.encoder.svtr_block.1.mlp.fc1.weight',
        'encoder.svtr_block.1.mlp.fc1.bias':
        'Student.head.ctc_encoder.encoder.svtr_block.1.mlp.fc1.bias',
        'encoder.svtr_block.1.mlp.fc2.weight':
        'Student.head.ctc_encoder.encoder.svtr_block.1.mlp.fc2.weight',
        'encoder.svtr_block.1.mlp.fc2.bias':
        'Student.head.ctc_encoder.encoder.svtr_block.1.mlp.fc2.bias',
        'encoder.norm.weight':
        'Student.head.ctc_encoder.encoder.norm.weight',
        'encoder.norm.bias':
        'Student.head.ctc_encoder.encoder.norm.bias',
        'encoder.conv3.conv.weight':
        'Student.head.ctc_encoder.encoder.conv3.conv.weight',
        'encoder.conv3.bn.weight':
        'Student.head.ctc_encoder.encoder.conv3.norm.weight',
        'encoder.conv3.bn.bias':
        'Student.head.ctc_encoder.encoder.conv3.norm.bias',
        'encoder.conv3.bn.running_mean':
        'Student.head.ctc_encoder.encoder.conv3.norm._mean',
        'encoder.conv3.bn.running_var':
        'Student.head.ctc_encoder.encoder.conv3.norm._variance',
        'encoder.conv4.conv.weight':
        'Student.head.ctc_encoder.encoder.conv4.conv.weight',
        'encoder.conv4.bn.weight':
        'Student.head.ctc_encoder.encoder.conv4.norm.weight',
        'encoder.conv4.bn.bias':
        'Student.head.ctc_encoder.encoder.conv4.norm.bias',
        'encoder.conv4.bn.running_mean':
        'Student.head.ctc_encoder.encoder.conv4.norm._mean',
        'encoder.conv4.bn.running_var':
        'Student.head.ctc_encoder.encoder.conv4.norm._variance',
        'encoder.conv1x1.conv.weight':
        'Student.head.ctc_encoder.encoder.conv1x1.conv.weight',
        'encoder.conv1x1.bn.weight':
        'Student.head.ctc_encoder.encoder.conv1x1.norm.weight',
        'encoder.conv1x1.bn.bias':
        'Student.head.ctc_encoder.encoder.conv1x1.norm.bias',
        'encoder.conv1x1.bn.running_mean':
        'Student.head.ctc_encoder.encoder.conv1x1.norm._mean',
        'encoder.conv1x1.bn.running_var':
        'Student.head.ctc_encoder.encoder.conv1x1.norm._variance',
    }
    for key in transpose:
        ppocr_state[key] = ppocr_state[key].transpose([1, 0])
    for key in torch_state:
        if key in state_map:
            torch_state[key] = torch.from_numpy(ppocr_state[state_map[key]])
    torch_state['decoder.decoder.weight'] = torch.from_numpy(
        ppocr_state['Student.head.ctc_head.fc.weight'].transpose(
            [1, 0]).reshape(6625, 64, 1, 1))
    torch_state['decoder.decoder.bias'] = torch.from_numpy(
        ppocr_state['Student.head.ctc_head.fc.bias'])

    return torch_state


def main():
    register_all_modules(init_default_scope=False)
    torch_model_path = '../../pretrained/ppocr_v3_svtr.pth'
    ppocr_model_path = '../../pretrained/ch_PP-OCRv3_rec_train/best_accuracy'
    cfg = Config.fromfile('../../configs/_base_/models/svtr_little.py')
    model = MODELS.build(cfg.model)

    ppocr_state = fluid.io.load_program_state(ppocr_model_path)

    state_torch = map_state(model.state_dict(), ppocr_state)
    torch.save(state_torch, torch_model_path)

    model.load_state_dict(state_torch)
    img = mmcv.imread('../../demo/3.jpg')
    cfg = Config.fromfile(
        '../../configs/_base_/datasets/rec_pipeline.py').test_pipeline
    cfg.pop(0)
    test_pipeline = Compose(cfg)
    inputs = test_pipeline({
        'img': img,
        'filename': 'test.jpg',
        'ori_filename': 'test.jpg',
        'ori_shape': img.shape,
        'img_shape': img.shape
    })
    inputs['inputs'] = [inputs['inputs']]
    inputs['data_samples'] = [inputs['data_samples']]
    model.eval()
    with torch.no_grad():
        y = model.decoder(model.encoder(model.backbone(model.data_preprocessor(inputs, True)['inputs'])))
        torch_outputs = torch.softmax(
            model.decoder(model.encoder(model.backbone(model.data_preprocessor(inputs, True)['inputs']))), -1)
    model_file_path = '../../pretrained/ch_PP-OCRv3_rec_infer/inference.pdmodel'
    params_file_path = '../../pretrained/ch_PP-OCRv3_rec_infer/inference.pdiparams'
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
    input_tensor.copy_from_cpu(model.data_preprocessor(inputs, True)['inputs'].numpy())
    predictor.run()
    pp_outputs = []
    for output_tensor in output_tensors:
        output = output_tensor.copy_to_cpu()
        pp_outputs.append(output)
    print(np.abs(pp_outputs[0] - torch_outputs.numpy()).mean())


if __name__ == '__main__':
    main()
