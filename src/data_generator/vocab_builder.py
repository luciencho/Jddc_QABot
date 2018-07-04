# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os
from src.utils import utils
from src.data_generator import vocabulary


def process(hparam):
    utils.raise_inexistence(hparam.tmp_dir)

    tokenizer = vocabulary.Tokenizer()
    all_data = []
    paths = [os.path.join(hparam.tmp_dir, 'train_q.txt'),
             os.path.join(hparam.tmp_dir, 'train_a.txt'),
             os.path.join(hparam.tmp_dir, 'dev_q.txt'),
             os.path.join(hparam.tmp_dir, 'dev_a.txt')]
    vocab_path = os.path.join(hparam.tmp_dir, '{}.vcb'.format(hparam.vocab_size))
    for path in paths:
        utils.raise_inexistence(path)
        all_data += utils.read_lines(path)
    tokenizer.build_vocab(all_data, hparam.vocab_size, vocab_path)
