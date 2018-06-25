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
import src.data_utils
from src.model import SoloModel


def train_shuffle(hparam, model, sess):
    # loading data
    word2id, _ = src.data_utils.load_vocab(hparam.vocab_path)
    train_question = src.data_utils.load_data(hparam.train_question_path)
    train_answer = src.data_utils.load_data(hparam.train_answer_path)
    train_batch = src.data_utils.Batch(
        train_question, train_answer, hparam.x_max_len, hparam.y_max_len, word2id)
    dev_question = src.data_utils.load_data(hparam.dev_question_path)
    dev_answer = src.data_utils.load_data(hparam.dev_answer_path)
    dev_batch = src.data_utils.Batch(
        dev_question, dev_answer, hparam.x_max_len, hparam.y_max_len, word2id)

    sess.run(tf.global_variables_initializer())
    starter = time()
    saver = tf.train.Saver(pad_step_number=True)
    train_losses = []
    dev_losses = []
    lowest_loss = 10
    train_id = 0
    dev_id = 0
    for i in range(hparam.max_iter):
        train_q, train_a, train_id, update_train_epoch = train_batch.next_batch(
            hparam.batch_size, train_id)
        train_fetch, train_feed_dict = model.step(train_q, train_a)
        _, train_loss = sess.run(train_fetch, feed_dict=train_feed_dict)
        train_losses.append(train_loss)
        if i % hparam.show_iter == 0 and i:
            dev_q, dev_a, dev_id, update_dev_epoch = dev_batch.next_batch(
                hparam.batch_size, dev_id)
            dev_fetch, dev_feed_dict = model.step(dev_q, dev_a, is_train=False)
            _, dev_loss = sess.run(dev_fetch, feed_dict=dev_feed_dict)
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
        if i in hparam.switch_iter:
            vecs = get_vector(hparam, model, sess)
            train_batch = src.data_utils.SimilarBatch(
                train_question, train_answer, hparam.x_max_len, hparam.y_max_len, word2id,
                vecs, 10, 4)
    print('lowest_loss: {:.5f}'.format(lowest_loss))


def get_vector(hparam, model, sess):
    saver = tf.train.Saver()
    saver.restore(sess, hparam.model_path)
    train_question = src.data_utils.load_data(hparam.train_question_path)
    train_answer = src.data_utils.load_data(hparam.train_answer_path)
    word2id, _ = src.data_utils.load_vocab(hparam.vocab_path)
    train_id = 0
    train_batch = src.data_utils.Batch(
        train_question, train_answer, hparam.x_max_len, hparam.y_max_len, word2id)
    total_size = len(train_question)
    question_vecs = []
    losses = []
    starter = time()
    for i in range(total_size // hparam.batch_size + (
            1 if total_size % hparam.batch_size != 0 else 0)):
        if i % hparam.show_iter == 0 and i:
            speed = hparam.show_iter / (time() - starter)
            print('step : {:05d} | speed: {:.5f} it/s'.format(i, speed))
            starter = time()
        train_q, train_a, train_id, update_train_epoch = train_batch.next_batch(
            hparam.batch_size, train_id)
        train_fetch, train_feed_dict = model.step(train_q, train_a, is_train=False)
        q, lo = sess.run(train_fetch, feed_dict=train_feed_dict)
        question_vecs.append(q)
        losses.append(lo)
    question_vecs = np.reshape(np.array(question_vecs), [-1, hparam.hidden])[: total_size]
    losses = losses[: total_size]
    print('total loss: {:.5f}'.format(sum(losses) / len(losses)))
    return question_vecs


def build_vector(hparam, model, sess):
    question_vecs = get_vector(hparam, model, sess)
    np.savez(hparam.vec_path, question_vecs=question_vecs)


def build_annoy(hparam, num_tree=10):
    vs = np.load(hparam.vec_path)
    qvs = vs['question_vecs']
    ann = AnnoyIndex(hparam.hidden)
    for n, ii in enumerate(qvs):
        ann.add_item(n, ii)
    ann.build(num_tree)
    ann.save(hparam.ann_path)


class Answer(object):
    def __init__(self, hparam):
        self.ann = AnnoyIndex(hparam.hidden)
        self.ann.load(hparam.ann_path)
        self.max_len = hparam.x_max_len
        self.answer = src.data_utils.load_data(hparam.train_answer_path)
        self.model = SoloModel(hparam)
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, hparam.model_path)
        self.word2id, _ = src.data_utils.load_vocab(hparam.vocab_path)

    def get_idx_by_item(self, n, num=5):
        return self.ann.get_nns_by_item(n, num)

    def get_idx_by_vec(self, vec, num=5):
        return self.ann.get_nns_by_vector(vec, num)

    def get_answer_by_vec(self, vec, num=5):
        idx = self.get_idx_by_vec(vec, num)
        return [self.answer[i] for i in idx]

    def get_answer_by_question(self, question, num=5):
        toks = src.data_utils.tokenizer(question, self.word2id, self.max_len)
        fetches, feed_dict = self.model.infer(toks)
        vec = self.sess.run(fetches, feed_dict=feed_dict)[0]
        return self.get_answer_by_vec(vec, num)
