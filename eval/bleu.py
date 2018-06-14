# -*- coding: utf-8 -*-
# @Time    : 6/14/18 12:57
# @Author  : Lucien Cho
# @File    : bleu.py
# @Software: PyCharm
# @Contact : luciencho@aliyun.com

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from collections import Counter
import math


def n_grams(sequence, n):
    sequence = iter(sequence)
    history = []
    while n > 1:
        history.append(next(sequence))
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]


def modified_precision(references, hypothesis, n, reference_weights):
    counts = Counter(n_grams(hypothesis, n)) if len(hypothesis) >= n else Counter()
    max_counts = {}
    for ref, weight in zip(references, reference_weights):
        ref_counts = Counter(n_grams(ref, n)) if len(ref) >= n else Counter()
        for gram, count in counts.items():
            max_counts[gram] = max(max_counts.get(gram, 0), weight * min(count, ref_counts[gram]))
    clipped_counts = {gram: min(count, max_counts[gram]) for gram, count in counts.items()}
    numerator = sum(clipped_counts.values())
    denominator = max(1, sum(counts.values()))
    return numerator / denominator


def closet_ref_len(len_refs, len_hyp):
    closet_ref_length = min(len_refs, key=lambda ref_len: (abs(ref_len - len_hyp), ref_len))
    return closet_ref_length


def brevity_penalty(len_refs, len_hyp):
    closet_ref_length = closet_ref_len(len_refs, len_hyp)
    if len_hyp > closet_ref_length:
        return 1
    elif not len_hyp:
        return 0
    else:
        return math.exp(1 - closet_ref_length / len_hyp)


def delta_bleu(references, hypothesis, gram_num=4, reference_weights=None, weight_range=None):
    num_ref = len(references)
    len_refs = [len(list(ref)) for ref in references]
    len_hyp = len(list(hypothesis))
    if reference_weights is None:
        reference_weights = [1] * num_ref
    elif weight_range is not None:
        reference_weights = normalized_weights(reference_weights, weight_range)
    s = []
    for n in range(gram_num):
        p_n = modified_precision(references, hypothesis, n + 1, reference_weights)
        try:
            s.append(math.log(p_n))
        except ValueError:
            s.append(0)
    bp = brevity_penalty(len_refs, len_hyp)
    s = bp * math.exp(math.fsum(s))
    return s


def normalized_weights(reference_weights, weight_range=(0, 10)):
    ws = []
    s0 = weight_range[0]
    s1 = weight_range[1]
    for w in reference_weights:
        if s0 < w < s1:
            w = (2 * w - s0 - s1) / (s1 - s0)
            ws.append(w)
        else:
            raise ValueError('{} is out of range {}-{}'.format(w, s0, s1))
    return ws
