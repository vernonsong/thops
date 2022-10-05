# @Time : 2022/7/20 15:53
# @Author : songweinan
# @Software: PyCharm
# 自能成羽翼，何必仰云梯。
# ==============================================================================
import os
import os.path as osp
import zipfile

INPUT_DIR = ''
OUTPUT_DIR = ''


def main():
    for file_name in os.listdir(INPUT_DIR):

        try:
            f = zipfile.ZipFile(osp.join(INPUT_DIR, file_name), 'r')
            for sub_file in f.namelist():
                f.extract(sub_file,
                          osp.join(OUTPUT_DIR,
                                   file_name.split('.')[0]))
            f.close()
        except Exception:
            print('Compressed package %s cannot be decompressed' % file_name)
            continue


main()
