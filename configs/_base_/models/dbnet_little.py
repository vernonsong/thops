# @Time : 2022/7/29 10:42
# @Author : songweinan
# @Software: PyCharm
# 自能成羽翼，何必仰云梯。
# ==============================================================================
model = dict(
    type='mmocr.DBNet',
    backbone=dict(
        type='thops.MobileNetV3',
        norm_cfg=dict(type='BN', eps=1e-05, momentum=0.1),
        arch='large',
        paddle_style=True,
        disable_se=True,
        out_indices=(3, 6, 12, 16),
    ),
    neck=dict(
        type='thops.RSEFPN',
        in_channels=[16, 24, 56, 480],
        out_channels=96,
        shortcut=True),
    det_head=dict(
        type='mmocr.DBHead',
        in_channels=96,
        module_loss=dict(type='DBModuleLoss'),
        postprocessor=dict(
            type='mmocr.DBPostprocessor',
            text_repr_type='quad',
            epsilon_ratio=0.001)),
    data_preprocessor=dict(
        type='mmocr.TextDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32))
