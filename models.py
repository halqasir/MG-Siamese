#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 13:03:51 2021

@author: hiba_alqasir
"""


from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.utils import plot_model
import tensorflow_addons as tfa
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import os

from config import cfg
from networks import create_net
from siamese import prep_model as prep_siamese_model
from siamese import contrastive_loss, eval_model_fixed_threshold


class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
                self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        print('[INFO] Learning rate is: {}'.format(K.eval(self.model.optimizer.lr)))


def lr_scheduler(epoch, lr):
    """
    Learning rate scheduler
    """
    if epoch == 700:
        return lr * 0.001
    return lr

def lr_scheduler_ganin(epoch, lr):
    """
    Learning rate scheduler
    """
    p = float(epoch) / cfg.epochs
    lr = 0.01 / (1. + 10 * p)**0.75

    return lr


def load_model():
    """
    Load model and weights
    """

    # load json and create model
    json_file = open('{}model.json'.format(cfg.MODEL_DIR), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    if cfg.siamese or cfg.pseudo_siamese:
        loss = contrastive_loss
    else:
        loss = cfg.loss
    loaded_model = model_from_json(loaded_model_json,
                custom_objects={
                    'InstanceNormalization':tfa.layers.InstanceNormalization, 
                    'LeakyReLU':layers.LeakyReLU, 'contrastive_loss': contrastive_loss})

    # loaded_model = prep_model(input_shape)

    # load weights into new model
    loaded_model.load_weights("{}model.h5".format(cfg.MODEL_DIR))

    print("[INFO] Model has been loaded from disk")
    # summarize model.
    loaded_model.summary()

    if cfg.optimizer == 'sdg':
        learning_rate = cfg.lr
        decay_rate = learning_rate / cfg.epochs
        opti = SGD(lr=learning_rate,
                   decay=decay_rate,
                   momentum=0.9, nesterov=True)
    elif cfg.optimizer == 'adam':
        learning_rate = cfg.lr
        opti = Adam(learning_rate=cfg.lr)
    else:
        opti = cfg.optimizer
    print("[INFO] Compiling model...")
    loaded_model.compile(loss=loss, optimizer=opti, metrics=["accuracy"])
    return loaded_model


def save_model(model):
    """
    Save model and weights
    """

    # serialize model to JSON
    model_json = model.to_json()
    with open("{}/model.json".format(cfg.MODEL_DIR), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("{}/model.h5".format(cfg.MODEL_DIR))
    print("[INFO] Model has been saved to the disk")


def learning_curves(history):
    """
    Plot training & validation accuracy & loss
    """

    acc = history.history['accuracy']
    loss = history.history['loss']

    try:
        val_acc = history.history['val_accuracy']
        val_loss = history.history['val_loss']
    except:
        val_acc = None
        val_loss = None

    # Plot training & validation accuracy values
    acc_fig = plt.figure()
    plt.title('Training Accuracy')
    plt.plot(acc, label='Training Accuracy')
    if val_acc is not None:
        plt.plot(val_acc, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    acc_fig.savefig('{}/accuracy.png'.format(cfg.MODEL_DIR), dpi=acc_fig.dpi)

    # Plot training & validation loss values
    loss_fig = plt.figure()
    plt.title('Training Loss')
    plt.plot(loss, label='Training Loss')
    if val_loss is not None:
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.xlabel('epoch')
    plt.ylim([0, 1.0])
    loss_fig.savefig('{}/loss.png'.format(cfg.MODEL_DIR), dpi=loss_fig.dpi)


def prep_model(input_shape):
    """
    Build and compile the model
    """

    if cfg.optimizer == 'sdg':
        learning_rate = cfg.lr
        decay_rate = learning_rate / cfg.epochs
        opti = SGD(lr=learning_rate,
                   decay=decay_rate,
                   momentum=0.9, nesterov=True)
    elif cfg.optimizer == 'adam':
        learning_rate = cfg.lr
        opti = Adam(learning_rate=cfg.lr)
    else:
        opti = cfg.optimizer

    if cfg.siamese or cfg.pseudo_siamese:
        model = prep_siamese_model(input_shape, opti)
    else:

        model = create_net(input_shape, cfg.net1)

        model.summary()
        print("[INFO] Compiling model...")
        model.compile(loss=cfg.loss, optimizer=opti,
                      metrics=["accuracy"])

    plot_model(model,
               to_file='{}/{}_model.png'.format(cfg.MODEL_DIR, cfg.net1),
               show_shapes=True, expand_nested=True, dpi=90)

    return model


def train_model(model, x_train, y_train, x_val, y_val):
    """
    Train the model
    """

    # save the loss to a file
    loss_history = LossHistory()

    _callbacks = [
        loss_history,
        callbacks.ModelCheckpoint(filepath='{}/tmp/weights.h5'.format(cfg.MODEL_DIR)),
        # callbacks.EarlyStopping(patience=2),
        # callbacks.TensorBoard(log_dir='./logs'),
        callbacks.LearningRateScheduler(lr_scheduler)
    ]

    if cfg.siamese or cfg.pseudo_siamese:
        if x_val is not None:
            valid_data = ([x_val[:, 0], x_val[:, 1]], y_val)
        else:
            valid_data = None
        history = model.fit([x_train[:, 0], x_train[:, 1]], y_train,
                            batch_size=cfg.batch_size,
                            epochs=cfg.epochs,
                            validation_data=valid_data,
                            callbacks=_callbacks)
    else:
        if x_val is not None:
            valid_data = (x_val, y_val)
        else:
            valid_data = None

        history = model.fit(x_train, y_train, batch_size=cfg.batch_size,
                            epochs=cfg.epochs, validation_data=valid_data,
                            callbacks=_callbacks)
    learning_curves(history)
    save_model(model)

    # save the loss to a file
    f = open('{}/loss_history.txt'.format(cfg.MODEL_DIR), 'w')
    for loss in loss_history.losses:
        f.write('{}\n'.format(loss))
    f.close()
    os.system('rm {}tmp/weights.h5'.format(cfg.MODEL_DIR))


def eval_model(model, x_test, y_test):
    """
    Evaluate the trained model
    """
    if cfg.siamese or cfg.pseudo_siamese:
        # loss, acc = model.evaluate([x_test[:, 0], x_test[:, 1]], y_test, verbose=0)
        acc = eval_model_fixed_threshold(model, x_test, y_test)
    else:
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
    return acc


def get_output_in_embed_space(model, x):
    for l in model.layers:
        print(l.name)

    if cfg.siamese:
        inner_model_name = model.layers[-2].name
        first_layer = model.get_layer(inner_model_name)
        last_layer = model.get_layer(inner_model_name).layers[-1]
        print(last_layer.name, first_layer.name, inner_model_name)
    else:
        first_layer = model.layers[0]
        last_layer = model.layers[-1]

    intermediate_layer_model = Model(inputs=first_layer.get_input_at(0),
                                     outputs=last_layer.get_output_at(0))

    plot_model(intermediate_layer_model, to_file='{}/intermediate_layer_model.png'.format(cfg.MODEL_DIR),
               show_shapes=True, expand_nested=True, dpi=90)

    intermediate_output = intermediate_layer_model.predict(x)

    print('[INFO] intermediate layer input shape: {}'.format(x.shape))
    print('[INFO] intermediate layer output shape: {}'.format(intermediate_output.shape))

    return intermediate_output
