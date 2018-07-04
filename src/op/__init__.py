# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function


import os
import tensorflow as tf
from collections import OrderedDict

from src.op import train, vectorize
from src.models import dual_encoder
from src.utils import problems
from src.utils import utils


registry_problem_fns = OrderedDict(
    train=[train, False],
    vectorize=[vectorize, False])


registry_models = dict(
    dual_encoder_rnn=dual_encoder.DualEncoderRNN,
    dual_encoder_cnn=dual_encoder.DualEncoderCNN)


def main(args):
    problem_fns = problems.parse_problems(args.problem, registry_problem_fns)
    model = problems.parse_model(args.model, registry_models)
    for n, (k, [fn, on_list]) in enumerate(problem_fns.items(), start=1):
        if on_list:
            utils.verbose('Start processing no.{} problem [{}]'.format(n, k))
            if args.gpu_device != '':
                os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory
            with tf.Session(config=config) as sess:
                fn.process(args, model(args), sess)
            utils.verbose('Finish processing no.{} problem [{}]'.format(n, k))
