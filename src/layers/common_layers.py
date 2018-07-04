# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function


import tensorflow as tf


_allowed_rnn_type = dict(
    lstm=tf.contrib.rnn.LSTMCell,
    gru=tf.contrib.rnn.GRUCell,
    rnn=tf.contrib.rnn.RNNCell)


def rnn_cell(hidden, num_layers=1, rnn_type='lstm', dropout=0.8, scope=None):

    def create_rnn_cell():
        cell = _allowed_rnn_type.get(rnn_type.lower(), 'rnn')(hidden, reuse=reuse)
        return tf.contrib.rnn.DropoutWrapper(cell, dropout)

    with tf.variable_scope(scope or 'rnn'):
        reuse = None if not tf.get_variable_scope().reuse else True
        return tf.contrib.rnn.MultiRNNCell(
            [create_rnn_cell() for _ in range(num_layers)], state_is_tuple=True)


def bidirectional_rnn(inputs, seq_lens, hidden, num_layers=1,
                      rnn_type='lstm', dropout=0.8, scope=None):
    with tf.variable_scope(scope or 'bd_rnn'):
        fw_cell = rnn_cell(hidden, num_layers, rnn_type, dropout, 'fw_cell')
        bw_cell = rnn_cell(hidden, num_layers, rnn_type, dropout, 'bw_cell')
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            fw_cell, bw_cell, inputs, seq_lens)
        return tf.concat(outputs, axis=-1), states[-1]


def length_last_axis(tensor):
    return tf.cast(
        tf.reduce_sum(tf.sign(tensor), axis=-1), dtype=tf.int32)


def rnn_attention(inputs, attention_size, return_alphas, name_scope=None):
    with tf.variable_scope('rnn_attention' or name_scope):
        hidden_size = inputs.shape[-1].value

        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

        with tf.name_scope('v'):
            # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
            #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

        if not return_alphas:
            return output
        else:
            return output, alphas
