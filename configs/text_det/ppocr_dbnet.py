# @Time : 2022/8/16 16:02
# @Author : songweinan
# @Software: PyCharm
# 自能成羽翼，何必仰云梯。
# ==============================================================================
_base_ = [
    '../_base_/datasets/det_pipeline.py',
    '../_base_/models/dbnet_little.py',
]

test_pipeline = _base_.test_pipeline_1333_736
st_det_test = dict(
    type='OCRDataset',
    data_root='',
    ann_file='instances_test.json',
    data_prefix=dict(img_path='imgs/'),
    test_mode=True,
    pipeline=test_pipeline)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=st_det_test)
