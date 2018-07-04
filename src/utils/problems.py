# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function


def parse_problems(problem, registry_problem_fns):
    problems = problem.lower().strip().split(',')
    for problem in problems:
        if problem not in registry_problem_fns:
            raise ValueError('problem {} is invalid'.format(problem))
        else:
            registry_problem_fns[problem][1] = True
    return registry_problem_fns


def parse_model(model, registry_models):
    model = model.lower().strip()
    if model not in registry_models:
        raise ValueError('model {} is invalid'.format(model))
    else:
        return registry_models[model]
