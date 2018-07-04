# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
from annoy import AnnoyIndex

from src.data_generator.data_loader import BaseBatch
from src.data_generator.vocabulary import Tokenizer
from src.utils import utils


def init_helper(hparam, question, answer):
    utils.raise_inexistence(question)
    utils.raise_inexistence(answer)
    return dict(
        input_x=utils.read_lines(question),
        input_y=utils.read_lines(answer),
        x_max_len=hparam.x_max_len,
        y_max_len=hparam.y_max_len)


def vector(hparam, model, sess):
    tokenizer = Tokenizer(hparam.vocab_file, segment=hparam.segment)
    batch = BaseBatch(
        tokenizer, init_helper(hparam, hparam.train_q, hparam.train_a))
    saver = tf.train.Saver()
    saver.restore(sess, hparam.model_path)
    return vector_space(hparam, model, sess, batch)


def vector_space(hparam, model, sess, batch):
    features = {'id': 0, 'vecs': []}
    total_size = batch.data_size
    starter = time.time()
    for i in range(total_size // hparam.batch_size + (
            1 if total_size % hparam.batch_size != 0 else 0)):
        fetches, feed_dict = model.infer_step(
            batch.next_batch(hparam.batch_size, features['id']))
        features['id'] += hparam.batch_size
        features['vecs'].append(sess.run(fetches, feed_dict=feed_dict))
        if i % hparam.show_steps == 0 and i:
            speed = hparam.show_steps / (time.time() - starter)
            utils.verbose('step : {:05d} | speed: {:.5f} it/s'.format(i, speed))
            starter = time.time()
    question_vecs = np.reshape(np.array(features['vecs']), [-1, hparam.hidden])[: total_size]
    vec_dim = question_vecs.shape[-1]
    ann = AnnoyIndex(vec_dim)
    for n, ii in enumerate(question_vecs):
        ann.add_item(n, ii)
    ann.build(10)
    return ann


def process(hparam, model, sess):
    ann = vector(hparam, model, sess)
    ann.save(hparam.ann_path)
    utils.verbose('dump annoy into {}'.format(hparam.ann_path))
