# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os
from src.hparams import solo_hparam
from src.utils import utils

registry_hparams = dict(
    solo_base=solo_hparam.solo_base(),
    solo_drop=solo_hparam.solo_tail(),
    solo_cnn=solo_hparam.solo_cnn())


def merge_hparam(args):
    if args.hparam_set not in registry_hparams:
        raise ValueError('invalid high parameter set {}'.format(args.hparam_set))
    else:
        hparam = registry_hparams[args.hparam_set]
        for k, v in hparam.__dict__.items():
            if not k.startswith('_'):
                utils.verbose('add attribute {} [{}] to hparams'.format(k, v))
                setattr(args, k, v)
    return args
