# @Time : 2022/8/16 15:44
# @Author : songweinan
# @Software: PyCharm
# 自能成羽翼，何必仰云梯。
# ======================================================================================================================
import torch
import torch.nn as nn

from thops.registry import MODELS


@MODELS.register_module()
class PaddleHSigmoid(nn.Module):
    """与mmcv的HSigmoid差别为需要增加一个1.2这个缩放因子."""

    def __init__(self,
                 bias: float = 3.0,
                 divisor: float = 6.0,
                 min_value: float = 0.0,
                 max_value: float = 1.0,
                 inplace=False):
        super().__init__()
        self.bias = bias
        self.divisor = divisor
        assert self.divisor != 0
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x * 1.2 + self.bias) / self.divisor
        return x.clamp_(self.min_value, self.max_value)
