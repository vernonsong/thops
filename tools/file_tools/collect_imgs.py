# @Time : 2022/7/20 15:38
# @Author : songweinan
# @Software: PyCharm
# 自能成羽翼，何必仰云梯。
# ==============================================================================

import argparse
import os
import os.path as osp
import shutil

import mmcv

from thops.utils import is_img

INPUT_DIR = ''
OUTPUT_DIR = ''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir',
        '-a',
        type=str,
        default=INPUT_DIR,
    )
    parser.add_argument('--output_dir', '-o', type=str, default=OUTPUT_DIR)
    args = parser.parse_args()
    return args


def traverse_dir(input_dir: str, output_dir: str):
    for name in os.listdir(input_dir):
        path = osp.join(input_dir, name)
        if osp.isdir(path):
            traverse_dir(path, output_dir)
        elif is_img(name):
            shutil.copy(path, osp.join(output_dir, name))


def main():
    args = parse_args()
    mmcv.mkdir_or_exist(args.output_dir)
    traverse_dir(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
