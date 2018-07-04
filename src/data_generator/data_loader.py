# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import random


class BaseBatch(object):
    def __init__(self, tokenizer, features):
        self.input_x = features['input_x']
        self.input_y = features['input_y']
        assert len(self.input_x) == len(self.input_y)
        self.x_max_len = features['x_max_len']
        self.y_max_len = features['y_max_len']
        self.tokenizer = tokenizer
        self.ann = None
        self.top_n = 1
        self.epoch = 1

    @property
    def data_size(self):
        return len(self.input_x)

    def shuffle_data(self):
        pair = list(zip(self.input_x, self.input_y))
        random.shuffle(pair)
        new_x, new_y = zip(*pair)
        self.input_x = new_x
        self.input_y = new_y

    def _next_ids(self, interval, idx):
        start = idx % self.data_size
        end = start % self.data_size + interval
        if end > self.data_size:
            ids = list(range(start, self.data_size)) + \
                  list(range(0, end - self.data_size))
            flag = True
        else:
            ids = list(range(start, end))
            flag = False
        return ids, flag

    def _next_batch_pairs(self, batch_size, idx, top_n):
        ids, update_epoch = self._next_ids(batch_size // self.top_n, idx)
        if top_n > 1:
            new_ids = []
            for i in ids:
                new_ids += self.most_similar_by_idx(i, top_n)
            ids = new_ids
        if update_epoch:
            self.epoch += 1
        pairs = [(self.input_x[i], self.input_y[i]) for i in ids]
        return pairs, update_epoch

    def _tokenize(self, line, max_len):
        tokens = self.tokenizer.encode_line(line)
        tokens = tokens[: max_len] + [self.tokenizer.PAD_ID] * (max_len - len(tokens))
        return tokens

    def next_batch(self, batch_size, idx):
        if batch_size % self.top_n != 0:
            raise ValueError('top_n {} must by a factor of batch_size {}'.format(
                self.top_n, batch_size))
        pairs, update_epoch = self._next_batch_pairs(batch_size, idx, self.top_n)
        input_x, input_y = zip(*pairs)
        features = dict(
            input_x=[self._tokenize(i, self.x_max_len) for i in input_x],
            input_y=[self._tokenize(i, self.y_max_len) for i in input_y],
            update_epoch=update_epoch,
            idx=idx + batch_size // self.top_n)
        return features

    def set_ann(self, ann, top_n):
        self.ann = ann
        self.top_n = top_n

    def most_similar_by_idx(self, idx, top_n):
        return self.ann.get_nns_by_item(idx, top_n)
