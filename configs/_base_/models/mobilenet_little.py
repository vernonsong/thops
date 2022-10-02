# @Time : 2022/8/20 07:29
# @Author : songweinan
# @Software: PyCharm
# 自能成羽翼，何必仰云梯。
# ======================================================================================================================
custom_imports = dict(
    imports=[
        'thops.models.backbones.mobilenet_enhance',
        'thops.models.backbones.mobilenet_v3',
        'thops.models.encoders.svtr_encoder', 'thops.models.recognizers.svtr'
    ],
    allow_failed_imports=False)
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='mmdet.MobileNetV3', widen_factor=0.35),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ),
)
