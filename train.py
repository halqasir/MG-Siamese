#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:34:42 2021

@author: hiba_alqasir
"""

from datasets import load_data
from models import prep_model, train_model, eval_model, load_model
from config import cfg, cfg_from_file

import pprint
import os
import time
import argparse
import sys
import numpy as np


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('--continue_training', dest='continue_training',
                        help='Continue training',
                        default=False, type=bool)
    parser.add_argument('--epochs', dest='epochs',
                        help='Number of epochs',
                        default=None, type=int)
    parser.add_argument('--lr', dest='lr',
                        help='Learning rate',
                        default=None, type=float)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    _args = parser.parse_args()
    return _args


if __name__ == '__main__':
    start_time = time.time()

    args = parse_args()
    print('[INFO] Called with args:')
    print(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lr is not None:
        cfg.lr = args.lr
    print('[INFO] Config:')
    pprint.pprint(cfg)

    if not os.path.exists(cfg.MODEL_DIR):
        os.mkdir(cfg.MODEL_DIR)

    pairs = False
    if cfg.siamese or cfg.pseudo_siamese:
        pairs = True


    if len(cfg.train_datasets) > 1:
        X_train = []
        Y_train = []
        for train_dataset in cfg.train_datasets:
            # input_shape, x_train, y_train, x_val, y_val = load_data(train_dataset, 'trainval', pairs)
            input_shape, x_train, y_train = load_data(train_dataset , 'train', pairs)
            for x, y in zip(x_train,y_train):
                X_train.append(x)
                Y_train.append(y)
        x_train = np.asarray(X_train)
        y_train = np.asarray(Y_train)

    else:
        input_shape, x_train, y_train = load_data(cfg.train_datasets[0] , 'train', pairs)

    x_val = None
    y_val = None



    if os.path.exists('{}/model.h5'.format(cfg.MODEL_DIR)):
        if args.continue_training: 
            model = load_model()
            os.system('cp {}/model.h5 {}/old_model.h5'.format(cfg.MODEL_DIR, cfg.MODEL_DIR))
            os.system('cp {}/accuracy.png {}/old_accuracy.png'.format(cfg.MODEL_DIR, cfg.MODEL_DIR))
            os.system('cp {}/loss.png {}/old_loss.png'.format(cfg.MODEL_DIR, cfg.MODEL_DIR))

        else:
            print('[INFO] Model is already trained') 
            print('[INFO] Execution time is {:.2f} minutes'.format((time.time() - start_time) / 60))
            sys.exit(0)
    else:
        if not os.path.exists('{}/tmp'.format(cfg.MODEL_DIR)):
            os.mkdir('{}/tmp'.format(cfg.MODEL_DIR))
        model = prep_model(input_shape)

    train_model(model, x_train, y_train, x_val, y_val)

    tr_acc = eval_model(model, x_train, y_train)
    print('[RES] Accuracy on training set: %0.2f%%' % (100 * tr_acc))

    if x_val is not None:
        val_acc = eval_model(model, x_val, y_val)
        print('[RES] Accuracy on validation set: %0.2f%%' % (100 * val_acc))

    print('[INFO] Execution time is {:.2f} minutes'.format((time.time() - start_time) / 60))
