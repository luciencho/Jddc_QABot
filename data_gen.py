# coding:utf-8
"""
generate data

python data_gen.py --tmp_dir ../tmp --data_dir ../dataset --problem data,vocab --hparam_set solo

"""
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import argparse
from src import data_generator, hparams


def parser_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--tmp_dir', type=str,
                        help='temporary directory')
    parser.add_argument('--data_dir', type=str,
                        help='data directory')
    parser.add_argument('--problem', type=str,
                        help='problems split by comma')
    parser.add_argument('--hparam_set', type=str,
                        help='high parameter set')
    args = parser.parse_args(args)
    args = hparams.merge_hparam(args)
    return args


if __name__ == '__main__':
    data_generator.main(parser_args())
