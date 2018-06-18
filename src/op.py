# -*- coding: utf-8 -*-
# @Time    : 6/10/18 11:14
# @Author  : Lucien Cho
# @File    : op.py
# @Software: PyCharm
# @Contact : luciencho@aliyun.com

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from annoy import AnnoyIndex
import tensorflow as tf
from time import time
import numpy as np
from src.data_utils import load_vocab, load_data, Batch, tokenizer
from src.model import SoloModel


def train(hparam, model, sess):
    # loading data
    word2id, _ = load_vocab(hparam.vocab_path)
    train_question = load_data(hparam.train_question_path)
    train_answer = load_data(hparam.train_answer_path)
    train_batch = Batch(hparam.batch_size, hparam.x_max_len, hparam.y_max_len,
                        train_question, train_answer, word2id)
    dev_question = load_data(hparam.dev_question_path)
    dev_answer = load_data(hparam.dev_answer_path)
    dev_batch = Batch(hparam.batch_size, hparam.x_max_len, hparam.y_max_len,
                      dev_question, dev_answer, word2id)

    sess.run(tf.global_variables_initializer())
    starter = time()
    saver = tf.train.Saver(pad_step_number=True)
    train_losses = []
    dev_losses = []
    lowest_loss = 10
    for i in range(hparam.max_iter):
        train_fetch, train_feed_dict = model.step(train_batch)
        _, train_loss = sess.run(train_fetch, feed_dict=train_feed_dict)
        train_losses.append(train_loss)
        if i % hparam.show_iter == 0 and i:
            dev_fetch, dev_feed_dict = model.step(dev_batch, is_train=False)
            _, __, dev_loss = sess.run(dev_fetch, feed_dict=dev_feed_dict)
            dev_losses.append(dev_loss)
            speed = hparam.show_iter / (time() - starter)
            print('\tstep {:05d} | train loss {:.5f} | dev loss {:.5f} | speed {:.5f} it/s'.format(
                i, train_loss, dev_loss, speed))
            starter = time()
        if i % hparam.save_iter == 0 and i:
            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_dev_loss = sum(dev_losses) / len(dev_losses)
            if avg_dev_loss < lowest_loss:
                saver.save(sess, hparam.model_path)
                lowest_loss = avg_dev_loss
            print('step {:05d} - {:05d} | avg train loss {:.5f} | avg dev loss {:.5f}'.format(
                i - hparam.save_iter, i, avg_train_loss, avg_dev_loss))
            train_losses = []
            dev_losses = []
    print('lowest_loss: {:.5f}'.format(lowest_loss))


def build_vector(hparam, model, sess):

    def annoy_with_npz(num_tree=10):
        vs = np.load(hparam.vec_path)
        qvs = vs['question_vecs']
        ann = AnnoyIndex(hparam.hidden)
        for n, ii in enumerate(qvs):
            ann.add_item(n, ii)
        ann.build(num_tree)
        ann.save(hparam.ann_path)

    saver = tf.train.Saver()
    saver.restore(sess, hparam.model_path)
    train_question = load_data(hparam.train_question_path)
    train_answer = load_data(hparam.train_answer_path)
    word2id, _ = load_vocab(hparam.vocab_path)
    train_batch = Batch(hparam.batch_size, hparam.x_max_len, hparam.y_max_len,
                        train_question, train_answer, word2id)
    total_size = len(train_question)
    question_vecs = []
    answer_vecs = []
    losses = []
    starter = time()
    for i in range(total_size // hparam.batch_size + (
            1 if total_size % hparam.batch_size != 0 else 0)):
        if i % hparam.show_iter == 0 and i:
            speed = hparam.show_iter / (time() - starter)
            print('step : {:05d} | speed: {:.5f} it/s'.format(i, speed))
            starter = time()
        train_fetch, train_feed_dict = model.step(train_batch, is_train=False)
        q, a, lo = sess.run(train_fetch, feed_dict=train_feed_dict)
        question_vecs.append(q)
        answer_vecs.append(a)
        losses.append(lo)
    question_vecs = np.reshape(np.array(question_vecs), [-1, hparam.hidden])[: total_size]
    answer_vecs = np.reshape(np.array(answer_vecs), [-1, hparam.hidden])[: total_size]
    losses = losses[: total_size]
    print('total loss: {:.5f}'.format(sum(losses) / len(losses)))
    np.savez(hparam.vec_path, question_vecs=question_vecs, answer_vecs=answer_vecs)
    annoy_with_npz()


class Answer(object):
    def __init__(self, hparam):
        self.ann = AnnoyIndex(hparam.hidden)
        self.ann.load(hparam.ann_path)
        self.max_len = hparam.x_max_len
        self.answer = load_data(hparam.train_answer_path)
        self.model = SoloModel(hparam)
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, hparam.model_path)
        self.word2id, _ = load_vocab(hparam.vocab_path)

    def get_idx_by_item(self, n, num=5):
        return self.ann.get_nns_by_item(n, num)

    def get_idx_by_vec(self, vec, num=5):
        return self.ann.get_nns_by_vector(vec, num)

    def get_answer_by_vec(self, vec, num=5):
        idx = self.get_idx_by_vec(vec, num)
        return [self.answer[i] for i in idx]

    def get_answer_by_question(self, question, num=5):
        toks = tokenizer(question, self.word2id, self.max_len)
        fetches, feed_dict = self.model.infer(toks)
        vec = self.sess.run(fetches, feed_dict=feed_dict)[0]
        return self.get_answer_by_vec(vec, num)
