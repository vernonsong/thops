# @Time : 2022/7/18 16:41
# @Author : songweinan
# @Software: PyCharm
# 自能成羽翼，何必仰云梯。
# ======================================================================================================================
import os.path as osp
from typing import Union

import mmcv
import numpy as np

from .img_utils import img2base64

__all__ = ['LabelMeWriter']


class LabelMeWriter(object):
    """用于生成labelme格式标注."""

    def __init__(self, img_path: str, version='4.5.6'):
        self.ann = {'version': version, 'flags': {}, 'shapes': []}
        img = mmcv.imread(img_path)
        self._outdir = osp.dirname(img_path)
        img_name = osp.basename(img_path)
        self._id = osp.splitext(img_name)[0]
        self.ann['imagePath'] = img_name
        self.ann['imageData'] = img2base64(img)
        self.ann['imageHeight'] = img.shape[0]
        self.ann['imageWidth'] = img.shape[1]

    def add_shape(self, points: Union[np.ndarray, list], label: str,
                  shape_type: str):
        points = np.array(points)
        if shape_type == 'rectangle' and points.shape != (2, 2):
            raise ValueError(
                'rectangle shape must be (2, 2) but get shape %s' %
                str(points.shape))
        if shape_type == 'polygon' and (points.shape[0] < 3
                                        or points.shape[1] != 2
                                        or len(points.shape) != 2):
            raise ValueError(
                'polygon shape must be (N, 2) and N > 2 but get shape %s' %
                str(points.shape))
        if shape_type == 'point' and points.shape != (1, 2):
            raise ValueError('point shape must be (1, 2) but get shape %s' %
                             str(points.shape))
        shape = {
            'label': label,
            'points': points,
            'group_id': None,
            'shape_type': shape_type,
            'flags': {}
        }
        self.ann['shapes'].append(shape)

    def save(self):
        mmcv.dump(self.ann, osp.join(self._outdir, self._id + '.json'), 'json')
