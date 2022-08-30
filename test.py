#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 10:30:10 2021

@author: hiba_alqasir
"""

from datasets import load_data, get_masks
from models import load_model, get_output_in_embed_space, eval_model
from utils import plot_tsne
from config import cfg, cfg_from_file
from siamese import get_dist

import numpy as np
import pprint
import os
import time
import argparse
import sys


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test network')
    parser.add_argument('--cfg', dest='cfg_file', 
                        help='optional config file',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    _args = parser.parse_args()
    return _args


def rot(out_angle):
    R = [0,15,30,45,60,75]
    X_masks = []
    Y_masks = []
        

    for r in R:
        if not r==out_angle:
           x_mask, y_mask = get_masks(r)
           for x,y in zip(x_mask,y_mask):
               X_masks.append(x)
               Y_masks.append(y)

    return X_masks, Y_masks


def accuracy(model, X, Y):
    if not (cfg.siamese or cfg.pseudo_siamese):
        acc = eval_model(model, X, Y)
    else:      
        x_mask, y_mask = rot(100)
        # x_mask, y_mask = get_masks(75)

        correct = 0
        for i in range(len(X)):
            x = X[i]
            y = np.argmax(Y[i])

            x = np.expand_dims(x, -1)
            x = np.moveaxis(x, -1, 0)
            D = []
            for m in x_mask:
                m = np.expand_dims(m, -1)
                m = np.moveaxis(m, -1, 0)
                d = get_dist(model, x, m)
                D.append(d[0,0])
            y_pred = np.argmin(D) % 10
            if y_pred == y:
                correct += 1
        acc = (1.0*correct)/len(X)
    return acc


if __name__ == '__main__':
    start_time = time.time()

    args = parse_args()
    print('[INFO] Called with args:')
    print(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    print('[INFO] Config:')
    pprint.pprint(cfg)

    if os.path.exists('{}/model.h5'.format(cfg.MODEL_DIR)):
        model = load_model()

        for test_set in cfg.test_datasets:     
            if test_set in cfg.train_datasets:
                x_test, y_test = load_data(test_set, 'test', False)
                x_source = x_test
                y_source = y_test
                te_acc = accuracy(model, x_test, y_test)
                print('[RES] Accuracy on {} test set: {:.2f}'.format(test_set, te_acc * 100))

        for test_set in cfg.test_datasets:
            if not test_set in cfg.train_datasets:
                x_test, y_test = load_data( test_set, 'test', False)
                te_acc = accuracy(model, x_test, y_test)
                print('[RES] Accuracy on {} test set: {:.2f}'.format(test_set, te_acc * 100))

                num_test = 200
                
                # combined_test_imgs = np.vstack([x_test[:num_test], x_source[:num_test]])
                # combined_test_labels = np.vstack([y_test[:num_test], y_source[:num_test]])
                # combined_test_domain = np.vstack([np.tile([1., 0.], [num_test, 1]),
                # np.tile([0., 1.], [num_test, 1])])
                
                # combined_test_emb = get_output_in_embed_space(model, combined_test_imgs)
                
                # plot_tsne(combined_test_emb, combined_test_labels, test_set, combined_test_domain, num_test)
    else:
        print('[ERROR] Model does not exist in {}'.format(cfg.MODEL_DIR))

    print('[INFO] Execution time is {:.2f} minutes'.format((time.time() - start_time) / 60))
