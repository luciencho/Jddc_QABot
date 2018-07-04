# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function


from collections import OrderedDict
from src.data_generator import data_builder, vocab_builder
from src.utils import problems
from src.utils import utils


registry_problem_fns = OrderedDict(
    data=[data_builder, False],
    vocab=[vocab_builder, False])


def main(args):
    problem_fns = problems.parse_problems(args.problem, registry_problem_fns)
    for n, (k, [fn, on_list]) in enumerate(problem_fns.items(), start=1):
        if on_list:
            utils.verbose('Start processing no.{} problem [{}]'.format(n, k))
            fn.process(args)
            utils.verbose('Finish processing no.{} problem [{}]'.format(n, k))
