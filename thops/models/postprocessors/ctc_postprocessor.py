# @Time : 2022/10/4 22:06
# @Author : songweinan
# @Software: PyCharm
# 自能成羽翼，何必仰云梯。
# ==============================================================================
from typing import Dict, Sequence, Union

from mmengine.registry import MODELS
from mmocr.models.common.dictionary import Dictionary
from mmocr.models.textrecog.postprocessors import \
    CTCPostProcessor as CTCPostProcessor_


@MODELS.register_module()
class CTCPostProcessor(CTCPostProcessor_):
    """mmocr的设计为最后将ignore token添加至末尾，ppocr为添加至开头."""

    def __init__(self,
                 dictionary: Union[Dictionary, Dict],
                 max_seq_len: int = 40,
                 ignore_chars: Sequence[str] = ['padding'],
                 **kwargs):
        super().__init__(
            dictionary=dictionary,
            max_seq_len=max_seq_len,
            ignore_chars=ignore_chars,
            **kwargs)
        self.ignore_indexes = [0]
