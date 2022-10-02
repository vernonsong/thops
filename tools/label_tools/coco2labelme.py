# @Time : 2022/7/18 17:34
# @Author : songweinan
# @Software: PyCharm
# 自能成羽翼，何必仰云梯。
# ======================================================================================================================
import argparse
import os.path as osp
import shutil

import mmcv
import numpy as np

from thops.utils import LabelMeWriter

ANN_PATH = ''
IMG_DIR = ''
OUTPUT_DIR = ''
SHAPE_TYPE = 'rectangle'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ann_path',
        '-a',
        type=str,
        default=ANN_PATH,
    )
    parser.add_argument('--img_dir', '-i', type=str, default=IMG_DIR)
    parser.add_argument('--output_dir', '-o', type=str, default=OUTPUT_DIR)
    parser.add_argument('--shape_type', '-s', type=str, default=SHAPE_TYPE)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    shape_type = args.shape_type
    mmcv.mkdir_or_exist(args.output_dir)
    ann = mmcv.load(args.ann_path)
    img_infos = ann['images']
    categories = ann['categories']
    annotation = ann['annotations']
    idx = 0
    for img_info in img_infos:
        file_name = img_info['file_name']
        img_id = img_info['id']
        shapes = []
        while idx < len(annotation) and annotation[idx]['image_id'] == img_id:
            if shape_type == 'rectangle':
                xmin = annotation[idx]['bbox'][0]
                ymin = annotation[idx]['bbox'][1]
                xmax = xmin + annotation[idx]['bbox'][2]
                ymax = ymin + annotation[idx]['bbox'][3]
                points = [[xmin, ymin], [xmax, ymax]]

            elif shape_type == 'polygon':
                points = np.array(annotation[idx]['segmentation']).reshape(
                    [-1, 2])
            else:
                idx += 1
                continue
            shapes.append({
                'points':
                points,
                'label':
                categories[annotation[idx]['category_id'] - 1]['name']
            })
            idx += 1

        shutil.copy(
            osp.join(args.img_dir, file_name),
            osp.join(args.output_dir, file_name))
        labelme_writer = LabelMeWriter(osp.join(args.output_dir, file_name))
        for shape in shapes:
            labelme_writer.add_shape(shape['points'], shape['label'],
                                     shape_type)
        labelme_writer.save()


if __name__ == '__main__':
    main()
