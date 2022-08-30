#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 13:01:42 2021

@author: hiba_alqasir
"""

from easydict import EasyDict
import numpy as np

__C = EasyDict()
cfg = __C

__C.test_datasets = ['usps', 'mnistm', 'svhn', 'mnist']
__C.train_datasets = ['mnist']
__C.im_size = 32

# General
__C.net1 = 'CNN2'
__C.batch_size = 64
__C.num_classes = 10
__C.epochs = 100
__C.lr = 0.00001
__C.loss = "categorical_crossentropy"
__C.optimizer = "sgd"
__C.with_masks = True
__C.handwritten_masks = False

__C.pseudo_siamese = False
__C.siamese = False

__C.tiny = True

__C.outdir = './outputs/baseline/'
__C.MODEL_DIR = '{}{}/{}/'.format(__C.outdir, __C.net1, __C.train_datasets)


def _merge_a_into_b(a, b):
    """
    Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not EasyDict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is EasyDict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """
    Load a config file and merge it into the default options.
    """
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = EasyDict(yaml.safe_load(f))

    _merge_a_into_b(yaml_cfg, __C)

    __C.MODEL_DIR = '{}{}_run2/LO_{}/'.format(__C.outdir, __C.net1, __C.test_datasets[0])

    # __C.MODEL_DIR = '{}{}/{}/'.format(__C.outdir, __C.net1, __C.train_datasets[0])
