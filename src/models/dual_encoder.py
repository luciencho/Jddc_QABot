# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
from src.layers import common_layers


class DualEncoderRNN(object):
    def __init__(self, hparam):
        self.hparam = hparam
        self.global_step = tf.Variable(0, trainable=False)
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=None, name='keep_prob')

        with tf.variable_scope('question'):
            self.question = tf.placeholder(tf.int32, [None, None], name='question')
            self.question_len = common_layers.length_last_axis(self.question)

        with tf.variable_scope('answer'):
            self.answer = tf.placeholder(tf.int32, [None, None], name='answer')
            self.answer_len = common_layers.length_last_axis(self.answer)

        with tf.variable_scope('labels'):
            self.labels = tf.placeholder(tf.int32, [None, None], name='labels')

        self.embedding = None
        self.emb_question = None
        self.emb_answer = None
        self.question_state = None
        self.answer_state = None
        self.probs = None
        self.acc = None
        self.show_loss = None
        self.mean_loss = None
        self.learning_rate = None
        self.opt = None
        self.optOp = None
        self.init = None
        self.bottom()
        self.body()
        self.top()

    def bottom(self):
        with tf.variable_scope('embedding'), tf.device('/cpu:0'):
            self.embedding = tf.get_variable(
                'embedding', [self.hparam.vocab_size, self.hparam.emb_dim],
                initializer=tf.contrib.layers.xavier_initializer())
            self.emb_question = tf.nn.embedding_lookup(self.embedding, self.question)
            self.emb_answer = tf.nn.embedding_lookup(self.embedding, self.answer)
            self.emb_question = tf.nn.dropout(
                self.emb_question, keep_prob=self.keep_prob)
            self.emb_answer = tf.nn.dropout(
                self.emb_answer, keep_prob=self.keep_prob)

    def body(self):
        fw_cell = common_layers.rnn_cell(
            self.hparam.hidden, self.hparam.num_layers,
            self.hparam.rnn_cell, self.keep_prob, scope='fw_cell')
        bw_cell = common_layers.rnn_cell(
            self.hparam.hidden, self.hparam.num_layers,
            self.hparam.rnn_cell, self.keep_prob, scope='bw_cell')
        with tf.variable_scope('rnn'):
            question_output, question_final_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_cell, cell_bw=bw_cell, inputs=self.emb_question,
                sequence_length=self.question_len, time_major=False, dtype=tf.float32)
            question_output = tf.concat([question_output[0], question_output[1]], axis=-1)
            question_final_state = question_final_state[-1]
            answer_output, answer_final_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_cell, cell_bw=bw_cell, inputs=self.emb_answer,
                sequence_length=self.answer_len, time_major=False, dtype=tf.float32)
            answer_output = tf.concat([answer_output[0], answer_output[1]], axis=-1)
            answer_final_state = answer_final_state[-1]

        if self.hparam.attention is None:
            self.question_state = question_final_state[-1].h  # [batch_size, hidden( * 2)]
            self.answer_state = answer_final_state[-1].h  # [batch_size, hidden( * 2)]
        elif self.hparam.attention == 'self_att':
            self.question_state = tf.nn.dropout(common_layers.rnn_attention(
                question_output, self.hparam.attention_size, False, 'question_attention'),
                self.hparam.keep_prob)
            self.answer_state = tf.nn.dropout(common_layers.rnn_attention(
                answer_output, self.hparam.attention_size, False, 'answer_attention'),
                self.hparam.keep_prob)
        else:
            raise ValueError('attention type {} is invalid'.format(self.hparam.attention))

    def top(self):
        w_shape = [self.answer_state.shape[-1]] * 2
        with tf.variable_scope('linear'):
            w = tf.get_variable(
                'linear_w', w_shape, initializer=tf.truncated_normal_initializer())
        logits = tf.matmul(
            self.question_state, tf.matmul(self.answer_state, w), transpose_b=True)
        losses = tf.losses.softmax_cross_entropy(self.labels, logits)
        self.probs = tf.nn.softmax(logits)
        self.acc = tf.contrib.metrics.accuracy(
            predictions=tf.argmax(logits, axis=-1), labels=tf.argmax(self.labels, axis=-1))
        self.show_loss = tf.reduce_mean(losses, name='show_loss')
        trainable_vars = tf.trainable_variables()
        self.mean_loss = tf.reduce_mean(
            losses + self.hparam.l2_weight * tf.add_n(
                [tf.nn.l2_loss(v) for v in trainable_vars if 'bias' not in v.name]),
            name='mean_loss')

        self.learning_rate = tf.train.exponential_decay(
            self.hparam.learning_rate, self.global_step, 100, self.hparam.decay_rate)
        self.opt = tf.contrib.opt.LazyAdamOptimizer(learning_rate=self.learning_rate)
        grads_vars = self.opt.compute_gradients(self.mean_loss)
        capped_grads_vars = [[
            tf.clip_by_value(g, -1, 1), v] for g, v in grads_vars if g is not None]
        self.optOp = self.opt.apply_gradients(capped_grads_vars, self.global_step)

    def train_step(self, features):
        feed_dict = {self.question: features['input_x'],
                     self.answer: features['input_y'],
                     self.labels: np.eye(len(features['input_x'])),
                     self.keep_prob: self.hparam.keep_prob}
        fetches = [self.optOp, self.show_loss, self.acc]
        return fetches, feed_dict

    def dev_step(self, features):
        feed_dict = {self.question: features['input_x'],
                     self.answer: features['input_y'],
                     self.labels: np.eye(len(features['input_x'])),
                     self.keep_prob: 1.0}
        fetches = [self.show_loss, self.acc]
        return fetches, feed_dict

    def infer_step(self, features):
        feed_dict = {self.question: features['input_x'],
                     self.keep_prob: 1.0}
        fetches = [self.question_state]
        return fetches, feed_dict


class DualEncoderCNN(DualEncoderRNN):
    def __init__(self, hparam):
        super(DualEncoderCNN, self).__init__(hparam)

    def conv_2d(self, embed_inputs, max_len):
        pooled_outputs = []
        for i, filter_size in enumerate(self.hparam.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, self.hparam.emb_dim, 1, self.hparam.num_filters]
                w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="w")
                b = tf.Variable(tf.constant(0.1, shape=[self.hparam.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    embed_inputs,
                    w,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, max_len - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        num_filters_total = self.hparam.num_filters * len(self.hparam.filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        out = tf.nn.dropout(h_pool_flat, self.keep_prob)
        return out

    def body(self):
        with tf.variable_scope('embedding_expanded'), tf.device('/cpu:0'):
            self.emb_question = tf.expand_dims(self.emb_question, -1)
            self.emb_answer = tf.expand_dims(self.emb_answer, -1)

        self.question_state = self.conv_2d(self.emb_question, self.hparam.x_max_len)
        self.answer_state = self.conv_2d(self.emb_answer, self.hparam.y_max_len)
