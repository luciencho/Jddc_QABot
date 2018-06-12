# -*- coding: utf-8 -*-
# @Time    : 6/12/18 23:58
# @Author  : Lucien Cho
# @File    : trainer.py
# @Software: PyCharm
# @Contact : luciencho@aliyun.com

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import argparse
import tensorflow as tf
from src.data_utils import add_dir_to_hparam
from src.op import train, build_vector
from src.model import SoloBase, SoloModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tmp_dir', type=str, help='temporary directory')
    args = parser.parse_args()
    hparam = SoloBase()
    hparam = add_dir_to_hparam(hparam, args.tmp_dir)
    model = SoloModel(hparam)
    with tf.Session() as sess:
        train(hparam, model, sess)
        build_vector(hparam, model, sess)


if __name__ == '__main__':
    main()
