# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os
import argparse

from src import op, hparams


def parser_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--tmp_dir', type=str,
                        help='temporary directory')
    parser.add_argument('--problem', type=str,
                        help='problems split by comma')
    parser.add_argument('--model', type=str,
                        help='model')
    parser.add_argument('--model_dir', type=str,
                        help='model directory')
    parser.add_argument('--hparam_set', type=str,
                        help='high parameter set')
    parser.add_argument('--gpu_device', type=str,
                        help='visible gpu devices')
    parser.add_argument('--gpu_memory', type=float, default=0.23,
                        help='gpu memory fraction')
    args = parser.parse_args(args)
    args = hparams.merge_hparam(args)
    args.vocab_file = os.path.join(args.tmp_dir, '{}.vcb'.format(args.vocab_size))
    args.train_q = os.path.join(args.tmp_dir, 'train_q.txt')
    args.train_a = os.path.join(args.tmp_dir, 'train_a.txt')
    args.dev_q = os.path.join(args.tmp_dir, 'dev_q.txt')
    args.dev_a = os.path.join(args.tmp_dir, 'dev_a.txt')
    args.ann_path = os.path.join(args.model_dir, 'vecs.ann')
    args.model_path = os.path.join(args.model_dir, 'model')
    return args


if __name__ == '__main__':
    op.main(parser_args())
