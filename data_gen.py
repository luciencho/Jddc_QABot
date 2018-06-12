# -*- coding: utf-8 -*-
# @Time    : 6/13/18 00:17
# @Author  : Lucien Cho
# @File    : data_gen.py
# @Software: PyCharm
# @Contact : luciencho@aliyun.com

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import argparse
from src.data_utils import add_dir_to_hparam
from src.model import SoloBase
from src import data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tmp_dir', type=str, help='temporary directory')
    parser.add_argument('-d', '--data_dir', type=str, help='data directory')
    args = parser.parse_args()
    hparam = SoloBase()
    hparam = add_dir_to_hparam(hparam, args.tmp_dir, args.data_dir)
    data.main(hparam)


if __name__ == '__main__':
    main()
