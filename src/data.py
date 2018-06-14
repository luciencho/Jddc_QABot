# -*- coding: utf-8 -*-
# @Time    : 6/7/18 22:32
# @Author  : Lucien Cho
# @File    : data.py
# @Software: PyCharm
# @Contact : luciencho@aliyun.com

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import os
import re
import random
import tqdm
from src.data_utils import sent2words, copy_head


def build_dialogues(chat_file):
    dialogues = []
    with open(chat_file, 'r', encoding='utf-8') as f:
        chat_data = f.read().split('\n')[1:]
        first_line = chat_data[0]
        words = first_line.strip().split('\t')
        prev_id = words[0]
        prev_sent_by = 'q' if words[2] == '0' else 'a'
        content = words[-1]
        dialogue = [(content, prev_sent_by)]
        for n, line in enumerate(chat_data[1:]):
            words = line.strip().split('\t')
            curr_id = words[0]
            sent_by = 'q' if words[2] == '0' else 'a'
            content = words[-1]
            if curr_id == prev_id:
                if sent_by == prev_sent_by:
                    dialogue[-1] = (dialogue[-1][0] + '\t' + content, sent_by)
                else:
                    dialogue.append((content, sent_by))
                if n == len(chat_data) - 1:
                    dialogues.append(dialogue)
            else:
                dialogues.append(dialogue)
                dialogue = [(content, sent_by)]
                prev_id = curr_id
            prev_sent_by = sent_by
    return dialogues


def split_dialogues(dialogues, train_dev_ratio=10):
    random.shuffle(dialogues)
    divider = int(len(dialogues) / train_dev_ratio)
    test_dialogues = dialogues[: divider]
    train_dialogues = dialogues[divider:]
    return train_dialogues, test_dialogues


def build_qa(dialogues, directory, prefix='train', mode='qaqaq'):
    q_path = os.path.join(directory, prefix + '_q.txt')
    a_path = os.path.join(directory, prefix + '_a.txt')
    counter = 0
    with open(q_path, 'w', encoding='utf-8') as fq:
        with open(a_path, 'w', encoding='utf-8') as fa:
            for dial in dialogues:
                content, sent_by = zip(*dial)
                full = ''.join(sent_by)
                for i in re.finditer(r'(?={})'.format(mode + 'a'), full):
                    question = '<s>'.join(content[i.start(): i.start() + len(mode)]) + '<s>'
                    answer = content[i.start() + len(mode)]
                    fq.write(question + '\n')
                    fa.write(answer + '\n')
                    counter += 1
                    if counter % 10000 == 0:
                        print('store {} lines for {} set'.format(counter, prefix))


def build_vocab(in_file, out_file, max_vocab_size=None):
    dictionary = dict()
    with open(in_file, 'r', encoding='utf-8') as f_in:
        for line in tqdm.tqdm(f_in):
            words = sent2words(line)
            for word in words:
                if word in dictionary:
                    dictionary[word] += 1
                else:
                    dictionary[word] = 1
    vocab_size = len(dictionary.values())
    print("{} words are in file: {}".format(vocab_size, in_file))
    words = sorted(dictionary, key=dictionary.get, reverse=True)
    if max_vocab_size is not None:
        if max_vocab_size - 4 < vocab_size:
            words = words[: max_vocab_size - 4]
    words = copy_head + words
    print("Total vocabulary size: {}".format(len(words)))
    with open(out_file, 'w', encoding='utf-8') as f_out:
        f_out.write('\n'.join(words))


def main(hparam):
    if not os.path.exists(hparam.tmp_dir):
        os.makedirs(hparam.tmp_dir)
    chat_file = os.path.join(hparam.data_dir, 'preliminaryData', 'chat.txt')
    dialogues = build_dialogues(chat_file)
    train_dialogues, dev_dialogues = split_dialogues(dialogues, 10)
    build_qa(train_dialogues, hparam.tmp_dir, 'train', 'qaqaq')
    build_qa(dev_dialogues, hparam.tmp_dir, 'dev', 'qaqaq')
    build_vocab(hparam.train_question_path, hparam.vocab_path, hparam.vocab_size)
