# @Time : 2022/7/18 16:42
# @Author : songweinan
# @Software: PyCharm
# 自能成羽翼，何必仰云梯。
# ======================================================================================================================
import base64
import os.path as osp
import uuid

import cv2
import mmcv
import numpy as np

__all__ = ['img2base64', 'is_img', 'buffer2img', 'debug_img']

IMAGE_SUFFIX = [
    'bmp', 'dib', 'png', 'jpg', 'jpeg', 'pbm', 'pgm'
    'ppm', 'tif', 'tiff'
]


def img2base64(img: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf8')


def base642img(base64_data: str) -> np.ndarray:
    pass


def is_img(path: str) -> bool:
    return True if osp.splitext(
        path)[-1].lower()[1:] in IMAGE_SUFFIX else False


def buffer2img(buffer: bytes) -> np.ndarray:
    img = cv2.imdecode(
        np.frombuffer(buffer, dtype='uint8'), cv2.IMREAD_UNCHANGED)
    return img


def debug_img(img: np.ndarray,
              name: str,
              is_binary=False,
              imshow=False,
              unique_path=False,
              output_dir='debug'):
    if is_binary:
        img = img * 255
    img = img.astype(np.uint8)
    if imshow:
        cv2.imshow(name, img)
        cv2.waitKey()
    img_name = name + '.jpg'

    if unique_path:
        img_name = str(uuid.uuid1()) + '.jpg'
        mmcv.imwrite(img, osp.join(osp.join(output_dir, name), img_name))
    else:
        mmcv.imwrite(img, osp.join(output_dir, img_name))
