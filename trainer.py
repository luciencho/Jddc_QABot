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

import os
import argparse
import tensorflow as tf
from src.data_utils import add_dir_to_hparam
import src.op as op
import src.model as mdl


_allowed_hparams = dict(
    solobase=mdl.SoloBase,
    solobibase=mdl.SoloBiBase,
    solobiatt=mdl.SoloBiAtt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tmp_dir', type=str, help='temporary directory')
    parser.add_argument('-g', '--gpu_device', type=str, default='',
                        help='visible gpu device')
    parser.add_argument('-m', '--memory_fraction', type=float, default=0.95,
                        help='gpu memory fraction')
    parser.add_argument('-s', '--hparam', type=str, default='solobase',
                        help='high parameter set')
    args = parser.parse_args()
    if args.gpu_device != '':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    assert args.hparam in _allowed_hparams
    hparam = _allowed_hparams[args.hparam]
    hparam = add_dir_to_hparam(hparam, args.tmp_dir)

    model = mdl.SoloModel(hparam)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.memory_fraction
    with tf.Session(config=config) as sess:
        op.train_shuffle(hparam, model, sess)
        op.build_vector(hparam, model, sess)
        op.build_annoy(hparam)


if __name__ == '__main__':
    main()
