# -*- coding: utf-8 -*-
# @Time : 2022/8/4 09:48
# @Author : songweinan
# @Software: PyCharm
# 自能成羽翼，何必仰云梯。
# ======================================================================================================================
img_norm_cfg = dict(
    mean=[0], std=[255])
test_pipeline = [
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img'])
        ])
]