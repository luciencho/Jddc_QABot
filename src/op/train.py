# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function


import time
import tensorflow as tf

from src.utils import utils
from src.data_generator.vocabulary import Tokenizer
from src.data_generator.data_loader import BaseBatch
from src.op.vectorize import vector_space
from src.op.vectorize import init_helper


def avg_features(features):
    for k, v in features.items():
        if isinstance(v, list):
            features[k] = sum(v) / len(v)
    return features


def reset_features(features):
    features['train_losses'] = []
    features['dev_losses'] = []
    features['train_acc'] = []
    features['dev_acc'] = []
    return features


def reorder_batch(hparam, model, sess, batch):
    ann = vector_space(hparam, model, sess, batch)
    batch.set_ann(ann, hparam.top_n)
    return batch


def process(hparam, model, sess):
    utils.clean_and_make_directory(hparam.model_dir)
    tokenizer = Tokenizer(hparam.vocab_file)
    train_batch = BaseBatch(
        tokenizer, init_helper(hparam, hparam.train_q, hparam.train_a))
    dev_batch = BaseBatch(
        tokenizer, init_helper(hparam, hparam.dev_q, hparam.dev_a))

    sess.run(tf.global_variables_initializer())
    starter = time.time()
    saver = tf.train.Saver(pad_step_number=True)
    features = {'lowest_loss': 10, 'train_id': 0, 'dev_id': 0}
    features = reset_features(features)

    for i in range(hparam.max_steps):
        train_batch_features = train_batch.next_batch(
            hparam.batch_size, features['train_id'])
        train_fetches, train_feed_dict = model.train_step(train_batch_features)
        features['train_id'] = train_batch_features['idx']
        _, train_loss, train_acc = sess.run(train_fetches, feed_dict=train_feed_dict)
        features['train_losses'].append(train_loss)
        features['train_acc'].append(train_acc)
        if i % hparam.show_steps == 0 and i:
            dev_fetches, dev_feed_dict = model.dev_step(
                dev_batch.next_batch(hparam.batch_size, features['dev_id']))
            features['dev_id'] += hparam.batch_size
            dev_loss, dev_acc = sess.run(dev_fetches, feed_dict=dev_feed_dict)
            features['dev_losses'].append(dev_loss)
            features['dev_acc'].append(dev_acc)
            speed = hparam.show_steps / (time.time() - starter)
            utils.verbose(
                r'        step {:05d} | train [{:.5f} {:.5f}] | '
                r'dev [{:.5f} {:.5f}] | speed {:.5f} it/s'.format(
                    i, train_loss, train_acc, dev_loss, dev_acc, speed))
            starter = time.time()

        if i % hparam.save_steps == 0 and i:
            features = avg_features(features)
            if features['dev_losses'] < features['lowest_loss']:
                saver.save(sess, hparam.model_path)
                features['lowest_loss'] = features['dev_losses']
            utils.verbose(
                r'step {:05d} - {:05d} | train [{:.5f} {:.5f}] | '
                r'dev [{:.5f} {:.5f}]'.format(
                    i - hparam.save_steps, i, features['train_losses'],
                    features['train_acc'], features['dev_losses'], features['dev_acc']))
            print('-+' * 55)
            features = reset_features(features)

        if train_batch_features['update_epoch']:
            train_batch.shuffle_data()
            if train_batch.epoch > 10:
                utils.verbose('update epoch and reorder data...')
                train_batch = reorder_batch(hparam, model, sess, train_batch)

    utils.write_result(hparam, features['lowest_loss'])
