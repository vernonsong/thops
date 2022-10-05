# @Time : 2022/10/5 12:44
# @Author : songweinan
# @Software: PyCharm
# 自能成羽翼，何必仰云梯。
# ==============================================================================
from mmocr.apis.inferencers import MMOCRInferencer

from thops.utils import register_all_modules


def main():
    register_all_modules(init_default_scope=False)
    ocr = MMOCRInferencer(
        det_config='../../configs/text_det/ppocr_dbnet.py',
        det_ckpt='../../pretrained/ppocr_v3_db.pth',
        rec_config='../../configs/text_rec/ppocr_svtr.py',
        rec_ckpt='../../pretrained/ppocr_v3_svtr.pth')
    out = ocr(['../../demo/2.jpg'], show=True, wait_time=1)
    print(out)


if __name__ == '__main__':
    main()
