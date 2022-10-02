# -*- coding: utf-8 -*-
# @Time : 2022/8/13 18:17
# @Author : songweinan
# @Software: PyCharm
# 自能成羽翼，何必仰云梯。
# ======================================================================================================================
dictionary = dict(
    type='Dictionary',
    dict_file='/Users/vernon/Downloads/paddle2torch_PPOCRv3-main/weights/ppocr_keys_v1.txt',
    with_padding=True,
with_unknown=True)
label_convertor = dict(
    type='CTCConvertor', dict_file='/Users/vernon/Downloads/paddle2torch_PPOCRv3-main/weights/ppocr_keys_v1.txt')
model = dict(
    type='thops.SVTR',
    backbone=dict(
        type='thops.MobileNetEnhanceBackbone',
        last_conv_stride=(1, 2)
        # norm_cfg=dict(type='BN', eps=1e-05, momentum=0.1),
      ),

    encoder=dict(type='thops.SVTREncoder', in_channels=512),
    decoder=dict(type='mmocr.CRNNDecoder', in_channels=64, rnn_flag=False,
                 postprocessor=dict(type='CTCPostProcessor'),
                 dictionary=dictionary),
data_preprocessor=dict(
        type='mmocr.TextRecogDataPreprocessor', mean=[127], std=[127])
    # loss=dict(type='CTCLoss'),
    # label_convertor=label_convertor,
)