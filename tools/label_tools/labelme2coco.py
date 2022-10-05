# @Time : 2022/7/19 10:07
# @Author : songweinan
# @Software: PyCharm
# 自能成羽翼，何必仰云梯。
# ==============================================================================
import argparse
import os
import os.path as osp
import shutil

import mmcv
import numpy as np

from thops.utils import is_img

INPUT_DIR = ''
CLASSES = []
OUTPUT_DIR = ''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-i', type=str, default=INPUT_DIR)
    parser.add_argument('--classes', '-c', type=str, default=CLASSES)
    parser.add_argument('--output_dir', '-o', type=str, default=OUTPUT_DIR)
    args = parser.parse_args()
    return args


def get_bbox(points):
    points = np.asarray(points)
    xmin = points[:, 0].min()
    ymin = points[:, 1].min()
    xmax = points[:, 0].max()
    ymax = points[:, 1].max()
    return [xmin, ymin, xmax - xmin, ymax - ymin]


def annotations_polygon(points, image_id, object_id, category_id):
    annotation = {}
    annotation['segmentation'] = [list(np.asarray(points).flatten())]
    annotation['iscrowd'] = 0
    annotation['image_id'] = image_id
    annotation['bbox'] = list(map(float, get_bbox(points)))
    annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
    annotation['category_id'] = category_id
    annotation['id'] = object_id
    return annotation


def annotations_rectangle(points, image_id, object_id, category_id):
    # 按逆时针排序
    (x1, y1), (x2, y2) = points
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    points = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
    annotation = {}
    annotation['segmentation'] = [list(np.asarray(points).flatten())]
    annotation['iscrowd'] = 0
    annotation['image_id'] = image_id
    annotation['bbox'] = list(
        map(float, [
            points[0][0], points[0][1], points[2][0] - points[0][0],
            points[2][1] - points[0][1]
        ]))
    annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
    annotation['category_id'] = category_id
    annotation['id'] = object_id
    return annotation


def parse_labelme_ann(ann: dict, image_id: int, object_id: int,
                      classes_map: dict) -> tuple:
    annotations_list = []
    shapes = ann['shapes']
    for shape in shapes:
        label = shape['label']
        # 不处理不在classes的类别实例
        if label not in classes_map:
            continue
        p_type = shape['shape_type']
        if p_type == 'polygon':
            annotations_list.append(
                annotations_polygon(shape['points'], image_id, object_id,
                                    classes_map[label]))
        elif p_type == 'rectangle':
            annotations_list.append(
                annotations_rectangle(shape['points'], image_id, object_id,
                                      classes_map[label]))
        object_id += 1

    img_info = {}
    img_info['height'] = ann['imageHeight']
    img_info['width'] = ann['imageWidth']
    img_info['id'] = image_id
    img_info['file_name'] = ann['imagePath'].split('/')[-1]

    return annotations_list, img_info, object_id


def get_categories(classes_map: dict):
    categories = []
    for label in classes_map:
        categories.append({
            'supercategory': 'component',
            'id': classes_map[label],
            'name': label
        })
    return categories


def main():
    args = parse_args()

    classes = [item for item in args.classes.split(',')] if isinstance(
        args.classes, str) else args.classes
    classes_map = dict(zip(classes, range(1, len(classes) + 1)))
    img_dir = osp.join(args.output_dir + '/images')
    mmcv.mkdir_or_exist(img_dir)
    image_id = 1
    object_id = 1
    annotations_list = []
    images_list = []
    for file_name in os.listdir(args.input_dir):
        if is_img(file_name):
            shutil.copyfile(
                osp.join(args.input_dir, file_name),
                osp.join(img_dir, file_name))
            file_id = os.path.splitext(file_name)[0]
            ann_path = osp.join(args.input_dir, file_id + '.json')
            ann_labelme = mmcv.load(ann_path)
            coco_ann_list, img_info, object_id = parse_labelme_ann(
                ann_labelme, image_id, object_id, classes_map)

            annotations_list += coco_ann_list
            images_list.append(img_info)

            image_id += 1

    coco_data = {
        'images': images_list,
        'categories': get_categories(classes_map),
        'annotations': annotations_list
    }
    mmcv.dump(coco_data, osp.join(OUTPUT_DIR + '/annotations',
                                  'instance.json'))


if __name__ == '__main__':
    main()
