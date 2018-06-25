# -*- coding: utf-8 -*-
# @Time    : 6/8/18 01:51
# @Author  : Lucien Cho
# @File    : data_utils.py
# @Software: PyCharm
# @Contact : luciencho@aliyun.com

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import os
import re
import random
import jieba_fast as jieba
from annoy import AnnoyIndex


def add_dir_to_hparam(hparam, tmp_dir, data_dir=None):
    hparam.tmp_dir = tmp_dir
    hparam.vocab_path = os.path.join(tmp_dir, 'train_q.vcb')
    hparam.train_question_path = os.path.join(tmp_dir, 'train_q.txt')
    hparam.train_answer_path = os.path.join(tmp_dir, 'train_a.txt')
    hparam.dev_question_path = os.path.join(tmp_dir, 'dev_q.txt')
    hparam.dev_answer_path = os.path.join(tmp_dir, 'dev_a.txt')
    hparam.model_path = os.path.join(tmp_dir, 'model.ckpt')
    hparam.vec_path = os.path.join(tmp_dir, 'result.vec.npz')
    hparam.ann_path = os.path.join(tmp_dir, 'question_vecs.ann')
    if data_dir is not None:
        hparam.data_dir = data_dir
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
    with open(path, 'r', encoding='utf-8') as f:
        vocab = f.read().split('\n')
    word2id = {w: n for n, w in enumerate(vocab)}
    id2word = {n: w for n, w in enumerate(vocab)}
    return word2id, id2word


def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().split('\n')


class Batch(object):
    def __init__(self, x, y, x_max_len, y_max_len, word2id):
        self.x = x
        self.y = y
        self.size = len(x)
        self.x_max_len = x_max_len
        self.y_max_len = y_max_len
        self.word2id = word2id

    def shuffle_data(self):
        pair = list(zip(self.x, self.y))
        random.shuffle(pair)
        x, y = zip(*pair)
        self.x = x
        self.y = y

    def next_batch_idx(self, batch_size, idx):
        start = idx % self.size
        end = start % self.size + batch_size
        if end > self.size:
            ids = list(range(start, self.size)) + list(range(0, end - self.size))
            flag = True
        else:
            ids = list(range(start, end))
            flag = False
        return ids, flag

    def encode_by_ids(self, ids):
        return ([tokenizer(self.x[i], self.word2id, self.x_max_len) for i in ids],
                [tokenizer(self.x[i], self.word2id, self.y_max_len) for i in ids])

    def next_batch(self, batch_size, idx):
        ids, update_epoch = self.next_batch_idx(batch_size, idx)
        if update_epoch:
            self.shuffle_data()
        x, y = self.encode_by_ids(ids)
        return x, y, idx + batch_size, update_epoch


class SimilarBatch(Batch):
    def __init__(self, x, y, x_max_len, y_max_len, word2id, vecs, num_trees, top_n):
        super(SimilarBatch, self).__init__(x, y, x_max_len, y_max_len, word2id)
        self.ann = None
        self.set_ann(x, y, vecs)
        self.top_n = top_n
        self.num_trees = num_trees

    def set_ann(self, x, y, vectors):
        self.x = x
        self.y = y
        self.ann = AnnoyIndex(vectors.shape[-1])
        for n, v in enumerate(vectors):
            self.ann.add_item(n, v)
        self.ann.build(self.num_trees)

    def most_similar_by_idx(self, idx):
        return self.ann.get_nns_by_item(idx, self.top_n)

    def next_batch(self, batch_size, idx):
        if batch_size % self.top_n != 0:
            raise ValueError('top_n {} must by a factor of batch_size {}'.format(
                self.top_n, batch_size))
        ids, update_epoch = self.next_batch_idx(batch_size / self.top_n, idx)
        new_ids = []
        for i in ids:
            new_ids += self.most_similar_by_idx(i)
        x, y = self.encode_by_ids(new_ids)
        return x, y, idx + batch_size / self.top_n, update_epoch
