# @Time : 2022/8/16 16:02
# @Author : songweinan
# @Software: PyCharm
# 自能成羽翼，何必仰云梯。
# ======================================================================================================================
_base_ = [
    '../_base_/datasets/det_pipeline.py',
    '../_base_/models/dbnet_little.py',
]
test_pipeline_1333_736 = {{_base_.test_pipeline_1333_736}}
data = dict(test=dict(pipeline=test_pipeline_1333_736))
