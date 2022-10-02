# @Time : 2022/7/19 20:03
# @Author : songweinan
# @Software: PyCharm
# 自能成羽翼，何必仰云梯。
# ======================================================================================================================
import os.path as osp

import fitz

__all__ = ['pdf2img']


def pdf2img(pdf_path: str, output_dir: str, prefix='', zoom=(3, 3)):
    pdf = fitz.open(pdf_path)
    for pg in range(0, pdf.page_count):
        page = pdf[pg]
        # 设置缩放和旋转系数
        if isinstance(zoom, int) or isinstance(zoom, float):
            zoom = (zoom, zoom)
        mat = fitz.Matrix(zoom[0], zoom[0])
        pm = page.get_pixmap(matrix=mat, alpha=False)
        # 开始写图像
        pm.save(osp.join(output_dir, prefix + str(pg)) + '.jpg')
