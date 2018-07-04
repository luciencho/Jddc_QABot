# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os
import time
import shutil


def verbose(line):
    print('[{}]\t{}'.format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), line))


def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def clean_and_make_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
        time.sleep(5)
    os.makedirs(directory)


def raise_inexistence(path):
    if not os.path.exists(path):
        raise ValueError('directory or path {} does not exist'.format(
            os.path.abspath(path)))


def read_lines(path):
    raise_inexistence(path)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip().split('\n')


def write_lines(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def write_result(args, loss):
    lines = []
    home_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'hparams')
    file_path = os.path.join(model_dir, '{}.{}'.format(int(time.time()), 'rst'))
    for k, v in args.__dict__.items():
        if not k.startswith('_'):
            lines.append('hparams.{}: [{}]'.format(k, v))
            verbose('hparams.{}: [{}]'.format(k, v))
    lines.append('lowest loss: [{}]'.format(loss))
    verbose('lowest loss: [{}]'.format(loss))
    write_lines(file_path, lines)
