#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: hiba_alqasir
"""

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.applications.resnet import ResNet50
from config import cfg


def create_net(input_shape, name):
    if name == 'cnn2':
        _model = create_cnn2_network(input_shape)
    if name == 'cnn3':
        _model = create_cnn3_network(input_shape)
    elif name == 'vgg16':
        _model = create_vgg16_network(input_shape)
    elif name == 'resnet50':
        _model = create_resnet50_network(input_shape)
    elif name == 'vada':
        _model = create_vada_network(input_shape)
    elif name == 'lenet5':
        _model = create_lenet5_network(input_shape)
    elif name == 'a':
        _model = create_a_network(input_shape)
    elif name == 'b':
        _model = create_b_network(input_shape)

    return _model


def create_lenet5_network(input_shape):
    """
        LeNet-5 architecture from lecun-98.
    """

    _input = layers.Input(shape=input_shape)

    x = layers.Conv2D(6, kernel_size=(5, 5), activation='relu')(_input)
    x = layers.Conv2D(16, kernel_size=(5, 5), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(120, activation='relu')(x)
    x = layers.Dense(84, activation='relu')(x)

    if cfg.siamese or cfg.pseudo_siamese:
         x = layers.Dense(cfg.num_classes)(x)
    else:
         x = layers.Dense(cfg.num_classes, activation = 'softmax')(x)

    return Model(_input, x)


def create_a_network(input_shape):
    """
    MNIST architecture from Ganin et al, output: num_classes
    """
    
    _input = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, kernel_size=(5, 5), activation='relu')(_input)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
    x = layers.Conv2D(48, kernel_size=(5, 5), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(100, activation = 'relu')(x)
    x = layers.Dense(100, activation = 'relu')(x)
    
    if cfg.siamese or cfg.pseudo_siamese:
         x = layers.Dense(cfg.num_classes)(x)
    else:
         x = layers.Dense(cfg.num_classes, activation = 'softmax')(x)

    return Model(_input, x)

def create_b_network(input_shape):
    """
    SNHN architecture from Ganin et al, output: num_classes
    """
    
    _input = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, kernel_size=(5, 5), activation='relu')(_input)
    x = layers.BatchNormalization(64)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2))(x)
    x = layers.Conv2D(64, kernel_size=(5, 5), activation='relu')(x)
    x = layers.BatchNormalization(64)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2))(x)
    x = layers.Conv2D(128, kernel_size=(5, 5), activation='relu')(x)
    x = layers.BatchNormalization(128)
    x = layers.Flatten()(x)
    x = layers.Dense(3072, activation = 'relu')(x)
    x = layers.BatchNormalization(3072)
    x = layers.Dense(2048, activation = 'relu')(x)
    x = layers.BatchNormalization(2048)

    if cfg.siamese or cfg.pseudo_siamese:
         x = layers.Dense(cfg.num_classes)(x)
    else:
         x = layers.Dense(cfg.num_classes, activation = 'softmax')(x)

    return Model(_input, x)




def create_vada_network(input_shape):
    """
    VADA a feature generator g, a feature classifier h that takes output of g as input
    """
    # import tensorlayer as tl
    import tensorflow_addons as tfa


    _input = layers.Input(shape=input_shape)

    # x = _input
    # x = tl.layers.InstanceNorm()(_input)
    x = tfa.layers.InstanceNormalization()(_input)
    x = layers.Conv2D(64, kernel_size=(3, 3), activation=layers.LeakyReLU(alpha=0.1))(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), activation=layers.LeakyReLU(alpha=0.1))(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), activation=layers.LeakyReLU(alpha=0.1))(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.GaussianNoise(1)(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), activation=layers.LeakyReLU(alpha=0.1))(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), activation=layers.LeakyReLU(alpha=0.1))(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), activation=layers.LeakyReLU(alpha=0.1))(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.GaussianNoise(1)(x)

    x = layers.Conv2D(64, kernel_size=(3, 3), activation=layers.LeakyReLU(alpha=0.1),padding='same')(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), activation=layers.LeakyReLU(alpha=0.1),padding='same')(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), activation=layers.LeakyReLU(alpha=0.1),padding='same')(x)


    if cfg.siamese  or cfg.pseudo_siamese:
        x = layers.Flatten()(x)
        # x = layers.Dense(576)(x)
        # x = layers.Dense(10)(x)

        # x = layers.GlobalAvgPool2D()(x)
        x = layers.Dense(cfg.num_classes)(x)

    else:
        x = layers.GlobalAvgPool2D()(x)
        x = layers.Dense(cfg.num_classes,  activation='softmax')(x)
    _model = Model(inputs=_input, outputs=x)

    return _model


def create_resnet50_network(input_shape):
    """
    ResNet50 conv part + FC4096 + FC1024 only FC are trainable 
    """
    _input = layers.Input(shape=input_shape)
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor= _input, pooling=None)
    for layer in base_model.layers:
        layer.trainable = False
        print('[INFO] Layer {} is trainable {}'.format(layer.name, layer.trainable))

    x = base_model.output
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dense(1024, activation='relu')(x)
    if cfg.siamese or cfg.pseudo_siamese:
        predictions = x
    else:
        predictions = layers.Dense(cfg.num_classes, activation='softmax')(x)
    _model = Model(inputs=base_model.input, outputs=predictions)

    return _model


def create_vgg16_network(input_shape):
    """
    VGG16 conv part + FC4096 + FC1024 only FC are trainable 
    """

    _input = layers.Input(shape=input_shape)
    base_model = VGG16(include_top=False, weights='imagenet', input_tensor=_input, pooling=None)
    
    for layer in base_model.layers:
        layer.trainable = False
        print('[INFO] Layer {} is trainable {}'.format(layer.name, layer.trainable))

    x = base_model.output
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dense(1024, activation='relu')(x)
    if cfg.siamese or cfg.pseudo_siamese:
        predictions = x
    else:
        predictions = layers.Dense(cfg.num_classes, activation='softmax')(x)
    _model = Model(inputs=base_model.input, outputs=predictions)

    return _model


def create_cnn2_network(input_shape):
    """
    CNN network, output: num_classes
    """
    
    _input = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(_input)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Flatten()(x)
    
    if cfg.siamese or cfg.pseudo_siamese:
         x = layers.Dense(cfg.num_classes)(x)
    else:
         x = layers.Dense(cfg.num_classes, activation = 'softmax')(x)

    return Model(_input, x)


def create_cnn3_network(input_shape):
    """
    CNN network , output: num_classes
    """
    
    _input = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(_input)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    if cfg.siamese or cfg.pseudo_siamese:
        x = layers.Dense(cfg.num_classes)(x)
    else:
        x = layers.Dense(cfg.num_classes, activation="softmax")(x)
        
    return Model(_input, x)
