# -*- coding: utf-8 -*-
# @Time : 2022/8/13 18:46
# @Author : songweinan
# @Software: PyCharm
# 自能成羽翼，何必仰云梯。
# ======================================================================================================================
from thops.registry import MODELS
from mmocr.models.textrecog.recognizers.encoder_decoder_recognizer import EncoderDecoderRecognizer


@MODELS.register_module()
class SVTR(EncoderDecoderRecognizer):
    pass
