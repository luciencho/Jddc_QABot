# -*- coding: utf-8 -*-
# @Time    : 6/14/18 15:12
# @Author  : Lucien Cho
# @File    : bleu.py
# @Software: PyCharm
# @Contact : luciencho@aliyun.com

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import argparse

from eval.bleu import delta_bleu
from src.data_utils import sent2words


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--answer_path', type=str, help='answer file path')
    parser.add_argument('-i', '--infer_path', type=str, help='inference file path')
    parser.add_argument('-r', '--result_path', type=str, help='result file path')
    parser.add_argument('-u', '--unit', type=str, default='char', help='segment unit')
    args = parser.parse_args()
    return args


def read_inferences(infer_path):
    with open(infer_path, 'r', encoding='utf-8') as f:
        text = f.read().strip().split('\n')
        assert len(text) == 50
    return iter(text)


def read_answers(answer_path):
    with open(answer_path, 'r', encoding='utf-8') as f:
        text = f.read().strip().split('\n\n')
        assert len(text) == 50
        references_tuple = []
        for answer in text:
            for line in answer.split('\n'):
                ref, weight = line.split('\t')
                references_tuple.append(tuple([ref, float(weight)]))
            refs, scs = zip(*references_tuple)
            references_tuple = [refs, scs]
            yield references_tuple
            references_tuple = []


def main():
    args = get_args()
    iter_answers = read_answers(args.answer_path)
    iter_inferences = read_inferences(args.infer_path)
    delta_bleu_scores = []
    if args.unit == 'char':
        fn = list
    elif args.unit == 'word':
        fn = sent2words
    else:
        raise ValueError('invalid unit: {}'.format(args.unit))
    for (r, w), h in zip(iter_answers, iter_inferences):
        r = [word for word in [fn(i) for i in r] if word != '_']
        h = [word for word in fn(h) if word != '_']
        s = delta_bleu(r, h, 4, w, (-8, 8))
        delta_bleu_scores.append(s)
    with open(args.result_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(map(str, delta_bleu_scores)))
    print('Final Delta Bleu: {}'.format(sum(delta_bleu_scores) / len(delta_bleu_scores)))


if __name__ == '__main__':
    main()
