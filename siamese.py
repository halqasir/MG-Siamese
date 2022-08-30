#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 11:05:46 2021

@author: hiba_alqasir
"""


import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

from networks import create_net
from config import cfg


def compute_accuracy(y_true, y_pred):
    """
    Compute classification accuracy with a fixed threshold on distances.
    https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py
    """
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    """
    Compute classification accuracy with a fixed threshold on distances.
    https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py
    """
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))
    
 
def euclidean_distance(vects):
    """
    Compute euclidean distance between two vectors.
    https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py
    """
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    """
    Contrastive loss from Hadsell-et-al.'06 :
     http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
     https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py
    """
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    a = (1 - y_true)
    x = K.mean(y_true * square_pred + a * margin_square)
    return x


def prep_model(input_shape, opti):
    """
    Build a Siamese or Peudo-Siamese network and compile the model
    """
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    if cfg.pseudo_siamese :
        base_network1 = create_net(input_shape,cfg.net1)
        base_network2 = create_net(input_shape,cfg.net2)
        
        processed_a = base_network1(input_a)
        processed_b = base_network2(input_b)
        
    elif cfg.siamese:
        base_network = create_net(input_shape,cfg.net1)
        
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)
        
        print('[INFO:] base network is')
        base_network.summary()
    else:
        print("You called the wrong the function")
        return

    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)
    
    print("[INFO] compiling model...")
    model.compile(loss=contrastive_loss,
                  optimizer=opti, 
                  metrics=[accuracy])
    
    return model


def get_dist(model, x, y):
    """
    Compute the distance in the embedding space
    """
    d = model.predict([x,y])
    return d


def eval_model_fixed_threshold(model, x_test, y_test):
    """
    Evaluate the trained model with a fixed threshold on distances.
    """
    y_pred = model.predict([x_test[:, 0], x_test[:, 1]])
    acc = compute_accuracy(y_test, y_pred)
    return acc
