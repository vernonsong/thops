# @Time : 2022/10/4 09:43
# @Author : songweinan
# @Software: PyCharm
# 自能成羽翼，何必仰云梯。
# ==============================================================================
from mmocr.models.textrecog.decoders import CRNNDecoder as CRNNDecoder_

from thops.registry import MODELS


@MODELS.register_module()
class CRNNDecoder(CRNNDecoder_):

    def forward_train(self, feat, out_enc, img_metas):
        return super(CRNNDecoder, self).forward_train(out_enc, out_enc,
                                                      img_metas)
