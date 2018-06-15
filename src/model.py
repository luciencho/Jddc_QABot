# -*- coding: utf-8 -*-
# @Time    : 6/8/18 17:21
# @Author  : Lucien Cho
# @File    : model.py
# @Software: PyCharm
# @Contact : luciencho@aliyun.com

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf


class ModelTemplate(object):
    def __init__(self, hparam):
        self.hparam = hparam
        self.global_step = tf.Variable(0, trainable=False)

    def body(self):
        raise NotImplementedError()

    def step(self, batch, is_train):
        raise NotImplementedError()

    def infer(self, question_toks):
        raise NotImplementedError()


class SoloModel(ModelTemplate):
    def __init__(self, hparam):
        super(SoloModel, self).__init__(hparam)
        self.keep_prob = None
        self.question = None
        self.question_len = None
        self.answer = None
        self.answer_len = None
        self.labels = None
        self.embedding = None
        self.emb_question = None
        self.emb_answer = None
        self.question_state = None
        self.answer_state = None
        self.mean_loss = None
        self.opt = None
        self.optOp = None
        self.init = None
        self.body()

    def body(self):

        def create_rnn_cell():
            if self.hparam.rnn_cell.lower() == 'lstm':
                cell = tf.contrib.rnn.LSTMCell(self.hparam.hidden)
            elif self.hparam.rnn_cell.lower() == 'gru':
                cell = tf.contrib.rnn.GRUCell(self.hparam.hidden)
            else:
                cell = tf.contrib.rnn.RNNCell(self.hparam.hidden)
            cell = tf.contrib.rnn.DropoutWrapper(
                cell, output_keep_prob=self.keep_prob)
            return cell

        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=None, name='keep_prob')

        with tf.variable_scope('question'):
            self.question = tf.placeholder(tf.int32, [None, None], name='question')
            self.question_len = tf.cast(
                tf.reduce_sum(tf.sign(self.question), axis=-1), dtype=tf.int32)

        with tf.variable_scope('answer'):
            self.answer = tf.placeholder(tf.int32, [None, None], name='answer')
            self.answer_len = tf.cast(
                tf.reduce_sum(tf.sign(self.answer), axis=-1), dtype=tf.int32)

        with tf.variable_scope('labels'):
            self.labels = tf.placeholder(tf.int32, [None, None], name='labels')

        with tf.variable_scope('embedding'), tf.device('/cpu:0'):
            self.embedding = tf.get_variable(
                'embedding', [self.hparam.vocab_size, self.hparam.emb_dim],
                initializer=tf.contrib.layers.xavier_initializer())  # [vocab_size, emb_dim]
            self.emb_question = tf.nn.embedding_lookup(self.embedding, self.question)
            self.emb_answer = tf.nn.embedding_lookup(self.embedding, self.answer)
            self.emb_question = tf.nn.dropout(
                self.emb_question, keep_prob=self.keep_prob)  # [batch_size, q_seq_len, emb_dim]
            self.emb_answer = tf.nn.dropout(
                self.emb_answer, keep_prob=self.keep_prob)  # [batch_size, a_seq_len, emb_dim]

        if self.hparam.direction == 'mono':
            with tf.variable_scope('fw_cell'):
                fw_cell = tf.contrib.rnn.MultiRNNCell(
                    [create_rnn_cell() for _ in range(self.hparam.num_layers)], state_is_tuple=True)
            with tf.variable_scope('rnn'):
                question_output, question_final_state = tf.nn.dynamic_rnn(
                    cell=fw_cell, inputs=self.emb_question,
                    sequence_length=self.question_len, time_major=False, dtype=tf.float32)

                answer_output, answer_final_state = tf.nn.dynamic_rnn(
                    cell=fw_cell, inputs=self.emb_answer,
                    sequence_length=self.answer_len, time_major=False, dtype=tf.float32)
        elif self.hparam.direction == 'bi':
            with tf.variable_scope('fw_cell'):
                fw_cell = tf.contrib.rnn.MultiRNNCell(
                    [create_rnn_cell() for _ in range(self.hparam.num_layers)], state_is_tuple=True)
            with tf.variable_scope('bw_cell'):
                bw_cell = tf.contrib.rnn.MultiRNNCell(
                    [create_rnn_cell() for _ in range(self.hparam.num_layers)], state_is_tuple=True)
            with tf.variable_scope('rnn'):
                question_output, question_final_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=fw_cell, cell_bw=bw_cell, inputs=self.emb_question,
                    sequence_length=self.question_len, time_major=False, dtype=tf.float32)
                question_final_state = question_final_state[-1]
                answer_output, answer_final_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=fw_cell, cell_bw=bw_cell, inputs=self.emb_answer,
                    sequence_length=self.answer_len, time_major=False, dtype=tf.float32)
                answer_final_state = answer_final_state[-1]
        else:
            raise ValueError()

        with tf.variable_scope('linear'):
            w = tf.get_variable('linear_w', [self.hparam.hidden, self.hparam.hidden],
                                initializer=tf.truncated_normal_initializer())

        answer_final_state = tf.matmul(answer_final_state[-1].h, w)
        self.question_state = question_final_state
        self.answer_state = answer_final_state
        logits = tf.matmul(
            question_final_state[-1].h, answer_final_state, transpose_b=True)
        losses = tf.losses.softmax_cross_entropy(self.labels, logits)
        self.mean_loss = tf.reduce_mean(losses, name='mean_loss')

        self.opt = tf.contrib.opt.LazyAdamOptimizer(learning_rate=self.hparam.learning_rate)
        grads_vars = self.opt.compute_gradients(self.mean_loss)
        capped_grads_vars = [[tf.clip_by_value(g, -1, 1), v] for g, v in grads_vars if g is not None]
        self.optOp = self.opt.apply_gradients(capped_grads_vars, self.global_step)

    def step(self, batch, is_train=True):
        question, answer = batch.question_answer_pair()
        feed_dict = {self.question: question,
                     self.answer: answer,
                     self.labels: np.eye(batch.size),
                     self.keep_prob: self.hparam.keep_prob}
        if is_train:
            fetches = [self.optOp, self.mean_loss]
        else:
            feed_dict[self.keep_prob] = 1.0
            fetches = [self.question_state, self.answer_state, self.mean_loss]
        return fetches, feed_dict

    def infer(self, question_toks):
        feed_dict = {self.question: [question_toks],
                     self.keep_prob: 1.0}
        fetches = [self.question_state]
        return fetches, feed_dict


class SoloBase(object):  # 4.23
    rnn_cell = 'lstm'
    hidden = 128
    keep_prob = 0.85
    num_layers = 1
    vocab_size = 2 ** 15
    emb_dim = 128
    learning_rate = 0.005
    max_iter = 5000
    show_iter = 100
    save_iter = 500
    batch_size = 256
    x_max_len = 128
    y_max_len = 32
    direction = 'mono'


class SoloBiBase(SoloBase):  # 3.75
    direction = 'bi'
    keep_prob = 0.75
