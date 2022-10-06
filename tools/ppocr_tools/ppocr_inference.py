# @Time : 2022/10/5 12:44
# @Author : songweinan
# @Software: PyCharm
# 自能成羽翼，何必仰云梯。
# ==============================================================================
import argparse
import os.path as osp

import matplotlib.pyplot as plt
from mmocr.apis.inferencers import MMOCRInferencer

from thops.utils import register_all_modules

# mac显示中文用
BASE_DIR = __file__[:__file__.find('/tools/ppocr_tools/ppocr_inference.py')]
INPUT_PATH = osp.join(BASE_DIR, 'demo/2.jpg')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-i', type=str, default=INPUT_PATH)
    args = parser.parse_args()
    return args


def main():
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    register_all_modules(init_default_scope=False)

    ocr = MMOCRInferencer(
        det_config=osp.join(BASE_DIR, 'configs/text_det/ppocr_dbnet.py'),
        det_ckpt=osp.join(BASE_DIR, 'pretrained/ppocr_v3_db.pth'),
        rec_config=osp.join(BASE_DIR, 'configs/text_rec/ppocr_svtr.py'),
        rec_ckpt=osp.join(BASE_DIR, 'pretrained/ppocr_v3_svtr.pth'))
    args = parse_args()
    ocr([args.input_path], show=True, wait_time=1, img_out_dir='out')


if __name__ == '__main__':
    main()
