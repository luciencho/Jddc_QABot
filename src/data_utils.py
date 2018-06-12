# -*- coding: utf-8 -*-
# @Time    : 6/8/18 01:51
# @Author  : Lucien Cho
# @File    : prepro.py
# @Software: PyCharm
# @Contact : luciencho@aliyun.com

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import os
import re
import codecs
import math
import jieba_fast as jieba


def add_dir_to_hparam(hparam, tmp_dir, data_dir=None):
    hparam.vocab_path = os.path.join(tmp_dir, 'train_q.vcb')
    hparam.train_question_path = os.path.join(tmp_dir, 'train_q.txt')
    hparam.train_answer_path = os.path.join(tmp_dir, 'train_a.txt')
    hparam.dev_question_path = os.path.join(tmp_dir, 'dev_q.txt')
    hparam.dev_answer_path = os.path.join(tmp_dir, 'dev_a.txt')
    hparam.model_path = os.path.join(tmp_dir, 'model.ckpt')
    hparam.vec_path = os.path.join(tmp_dir, 'result.vec.npz')
    hparam.ann_path = os.path.join(tmp_dir, 'question_vecs.ann')
    if data_dir is not None:
        hparam.chat_file = os.path.join(data_dir, 'preliminaryData', 'chat.txt')
    return hparam


_substitution = [
    (re.compile(r'\[ORDERID_\d+\]'), '_ORDID_'),
    (re.compile(r'#E-s\d*\[数字x\]'), '_EMOJI_'),
    (re.compile(r'\[邮箱x\]'), '_EMAIL_'),
    (re.compile(r'\[数字x\]'), '_NUMB_'),
    (re.compile(r'\[地址x\]'), '_LOCA_'),
    (re.compile(r'\[时间x\]'), '_TIME_'),
    (re.compile(r'\[日期x\]'), '_DATE_'),
    (re.compile(r'\[链接x\]'), '_LINK_'),
    (re.compile(r'\[电话x\]'), '_PHON_'),
    (re.compile(r'\[金额x\]'), '_PRIC_'),
    (re.compile(r'\[姓名x\]'), '_NAME_'),
    (re.compile(r'\[站点x\]'), '_STAT_'),
    (re.compile(r'\[身份证号x\]'), '_PSID_'),
    (re.compile(r'\[组织机构x\]'), '_ORGN_'),
    (re.compile(r'\[子\]'), ''),
    (re.compile(r'\[父原始\]'), ''),
    (re.compile(r'\[父\]'), ''),
    (re.compile(r'~O\(∩_∩\)O/~'), '_SMIL_'),
    (re.compile(r'<s>'), '_SSS_'),
    (re.compile(r'\d{6,10}'), '_NUMB_'),
    (re.compile(r'\d{11}'), '_PHON_'),
    (re.compile(r'\d{11,15}'), '_NUMB_'),
]
copy_head = ['PAD', 'UNK', 'GO', 'EOS']


def clean_line(line):
    for regex, subs in _substitution:
        line = regex.sub(subs, line)
    return line.strip()


def sent2words(line):
    line = clean_line(line)
    words = jieba.lcut(line)
    return words


def tokenizer(line, word2id, max_len=None):
    words = sent2words(line)
    toks = [word2id[word] if word in word2id else word2id['UNK'] for word in words]
    if max_len is not None:
        toks = toks[-max_len:] + [word2id['UNK']] * (max_len - len(toks))
    return toks


def load_vocab(path):
    with codecs.open(path, 'r', 'utf-8') as f:
        vocab = f.read().split('\n')
    word2id = {w: n for n, w in enumerate(vocab)}
    id2word = {n: w for n, w in enumerate(vocab)}
    return word2id, id2word


def load_data(path):
    with codecs.open(path, 'r', 'utf-8') as f:
        return f.read().split('\n')


def iter_batch(lines, batch_size, word2id, max_len=None):
    start = 0
    size = len(lines)
    while True:
        start = start % size
        end = start % size + batch_size
        if end > size:
            curr = lines[start:] + lines[: end - size]
        else:
            curr = lines[start: end]
        curr = [tokenizer(l, word2id, max_len) for l in curr]
        assert len(curr) == batch_size
        yield curr
        start += batch_size


class Batch(object):
    def __init__(self, batch_size, x_max_len, y_max_len,
                 x, y, word2id, reverse_x=True):
        assert len(x) == len(y)
        print('data size: {}'.format(len(x)))
        self.size = batch_size
        self.x_max_len = x_max_len
        self.y_max_len = y_max_len
        if reverse_x:
            x = [i[:: -1] for i in x]
        self.x_iter = iter_batch(x, batch_size, word2id, x_max_len)
        self.y_iter = iter_batch(x, batch_size, word2id, y_max_len)

    def question(self):
        return next(self.x_iter)

    def answer(self):
        return next(self.y_iter)


def cosine_similarity(v1, v2):
    sum_xx, sum_xy, sum_yy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]
        y = v2[i]
        sum_xx += x * x
        sum_yy += y * y
        sum_xy += x * y
    return sum_xy / math.sqrt(sum_xx * sum_yy)\
