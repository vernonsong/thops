# @Time : 2022/8/13 18:46
# @Author : songweinan
# @Software: PyCharm
# 自能成羽翼，何必仰云梯。
# ======================================================================================================================
from mmocr.models.textrecog.recognizers.encoder_decoder_recognizer import \
    EncoderDecoderRecognizer

from thops.registry import MODELS


@MODELS.register_module()
class SVTR(EncoderDecoderRecognizer):
    pass
