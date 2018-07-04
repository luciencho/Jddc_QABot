# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function


import tensorflow as tf


def solo_base():
    hparams = tf.contrib.training.HParams(
        vocab_size=50000)
    hparams.rnn_cell = 'lstm'
    hparams.hidden = 128
    hparams.keep_prob = 0.75
    hparams.num_layers = 1
    hparams.emb_dim = 256
    hparams.learning_rate = 0.005
    hparams.max_steps = 30000
    hparams.show_steps = 50
    hparams.save_steps = 250
    hparams.batch_size = 256
    hparams.x_max_len = 128
    hparams.y_max_len = 32
    hparams.l2_weight = 0.0001
    hparams.attention = None
    hparams.attention_size = 32
    hparams.decay_rate = 0.97
    hparams.top_n = 4
    hparams.segment = 'jieba'
    return hparams


def solo_rnn():  # 3.336, 0.191
    hparams = solo_base()
    hparams.decay_rate = 0.98
    hparams.keep_prob = 0.7
    hparams.top_n = 16
    return hparams


def solo_thu():
    hparams = solo_rnn()
    hparams.segment = 'thulac'
    return hparams


def solo_cnn():
    hparams = solo_base()
    hparams.learning_rate = 0.002
    hparams.num_filters = 128
    hparams.filter_sizes = [3, 4, 5]
    hparams.decay_rate = 0.98
    hparams.keep_prob = 0.7
    hparams.top_n = 16
    return hparams
