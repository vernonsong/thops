# @Time : 2022/8/13 17:11
# @Author : songweinan
# @Software: PyCharm
# 自能成羽翼，何必仰云梯。
# ======================================================================================================================

img_norm_cfg = dict(mean=[127], std=[127])
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='mmocr.RescaleToHeight',
        height=48,
        min_width=48,
        max_width=None,
        width_divisor=16),
    # dict(type='Normalize', **img_norm_cfg),
    dict(
        type='mmocr.PackTextRecogInputs',
        meta_keys=('ori_shape', 'img_shape', 'valid_ratio'))
]
