# -*- coding: utf-8 -*-
# @Time : 2022/8/1 09:21
# @Author : songweinan
# @Software: PyCharm
# 自能成羽翼，何必仰云梯。
# ======================================================================================================================
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

test_pipeline_1333_736 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='Resize', scale=(1333, 736), keep_ratio=True),
    dict(
        type='mmocr.PackTextDetInputs',
        meta_keys=('ori_shape', 'img_shape', 'scale_factor'))
]