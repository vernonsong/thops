# -*- coding: utf-8 -*-
# @Time : 2022/8/20 16:19
# @Author : songweinan
# @Software: PyCharm
# 自能成羽翼，何必仰云梯。
# ======================================================================================================================
_base_ = [
    '../_base_/datasets/rec_pipeline.py',
    '../_base_/models/svtr_little.py',
]
test_pipeline = {{_base_.test_pipeline}}
data = dict(
    test=dict(pipeline=test_pipeline)
)
custom_imports = dict(imports=['models.backbones.mobilenet_enhance',
                               'models.backbones.mobilenet_v3',
                               'models.encoders.svtr_encoder',
                               'models.recognizers.svtr',
                               'models.decoders.crnn_decoder',
                               'mmocr.datasets'], allow_failed_imports=False)