#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 12:34:42 2021

@author: hiba_alqasir
"""

import numpy as np
import tensorflow.keras as keras
import pickle as pkl
import cv2
import h5py

from skimage.transform import resize
from scipy import ndimage
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from utils import imshow_grid
from config import cfg


def load_data(dataset, mode, pairs=False):
    """
    Load dataset train+validation set or test set
    """

    x_mask, y_mask = load_masks(cfg.handwritten_masks)
    
    if dataset == 'mnistm':
        x_train, y_train, x_test, y_test = load_mnistm()
    elif dataset == 'mnist':
        # x_train, y_train, x_test, y_test = load_mnist()
        x_train, y_train, x_test, y_test = load_rotated_mnist(0)
        x_mask = rotate_masks(x_mask , 0)
    elif dataset == 'svhn':
        x_train, y_train, x_test, y_test = load_svhn()
    elif dataset == 'usps':
        x_train, y_train, x_test, y_test = load_usps()
    elif dataset == 'mnist-r15':
        x_train, y_train, x_test, y_test = load_rotated_mnist(15)
        x_mask = rotate_masks(x_mask , 15)
    elif dataset == 'mnist-r30':
        x_train, y_train, x_test, y_test = load_rotated_mnist(30)
        x_mask = rotate_masks(x_mask , 30)
    elif dataset == 'mnist-r45':
        x_train, y_train, x_test, y_test = load_rotated_mnist(45)
        x_mask = rotate_masks(x_mask , 45)
    elif dataset == 'mnist-r60':
        x_train, y_train, x_test, y_test = load_rotated_mnist(60)
        x_mask = rotate_masks(x_mask , 60)
    elif dataset == 'mnist-r75':
        x_train, y_train, x_test, y_test = load_rotated_mnist(75)
        x_mask = rotate_masks(x_mask , 75)
    else:
        print('[ERROR] We do not have a dataset called {}'.format(dataset))
        return

    if cfg.tiny:
        x_train, y_train = get_tiny_dataset(x_train, y_train, max=100)

    x_mask, y_mask = type_size_scale(x_mask, y_mask)
    x_train, y_train = type_size_scale(x_train, y_train)
    x_test, y_test = type_size_scale(x_test, y_test)

    im_shape = x_train.shape[1:]
    imshow_grid(x_train, '{}'.format(dataset))

    if pairs:
        imshow_grid(x_mask, '{}_masks'.format(dataset))
        if mode == 'train' or mode == 'trainval':
            print("m {},   t {}".format(x_mask.shape, x_train.shape))
            x_train, y_train = create_all_pairs(x_train, y_train,
                                                x_mask, y_mask)
        elif mode == 'test':
            x_test, y_test = create_all_pairs(x_test, y_test,
                                              x_mask, y_mask)
    else:
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, cfg.num_classes)
        y_test = keras.utils.to_categorical(y_test, cfg.num_classes)

    if mode == 'trainval':
        # Split train data into train and validation sets
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                          test_size=0.15,
                                                          random_state=22)
    if cfg.with_masks:
        imshow_grid(x_mask, 'Masks')
        y_mask = keras.utils.to_categorical(y_mask, cfg.num_classes)
        x_train = np.concatenate([x_train, x_mask])
        y_train = np.concatenate([y_train, y_mask])

    if mode == 'trainval':
        print('[INFO]  x_train shape: {}'.format(x_train.shape))
        print('[INFO] {} train samples'.format(x_train.shape[0]))
        print('[INFO] {} validation samples'.format(x_val.shape[0]))
        return im_shape, x_train, y_train, x_val, y_val
    elif mode == 'train':
        print('[INFO]  x_train shape: {}'.format(x_train.shape))
        print('[INFO] {} train samples'.format(x_train.shape[0]))
        return im_shape, x_train, y_train
    elif mode == 'test':
        print('[INFO] {} test samples'.format(x_test.shape[0]))
        return x_test, y_test


def type_size_scale(x, y):
    # Convert images into 'float64' type
    x = x.astype('float64')

    # Convert labels into 'float64' type    
    y = y.astype('float64')

    # Resize images
    if x.shape[1] != cfg.im_size:
        x = resize_images(x)

    # Normalize the images data
    if x.max() != 1.0:
        print('[INFO] Scale images from [{}: {}] to [0.0:1.0]'.format(x.min(), x.max()))
        x /= x.max()

    return x, y


def load_mnist():
    """
    Prepare MNIST data
    """
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    # Make sure images have shape (28, 28, 3)
    x_train = np.concatenate([x_train, x_train, x_train], 3)
    x_test = np.concatenate([x_test, x_test, x_test], 3)

    return x_train, y_train, x_test, y_test


def load_rotated_mnist(angle):
    """
    Prepare rotated MNIST data
    """
    # x_train, y_train, x_test, y_test = load_mnist()
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)


    rx_train = []
    rx_test = []

    for x in x_train:
        new_img = ndimage.rotate(x, angle, reshape=False)
        rx_train.append(new_img)

    rx_train = np.asarray(rx_train)

    for x in x_test:
        new_img = ndimage.rotate(x, angle, reshape=False)
        rx_test.append(new_img)
    rx_test = np.asarray(rx_test)

    # print("RRRRRR {}".format(rx_test.shape))

    return rx_train, y_train, rx_test, y_test


def rotate_masks(x_mask , r):
    """
    Prepare rotated masks
    """
    rx_mask = []
    for x in x_mask:
        new_img = ndimage.rotate(x, r, reshape=False) # , order=0)
        rx_mask.append(new_img)
    rx_mask = np.asarray(rx_mask)

    return rx_mask


def load_mnistm():
    """
    Prepare MNISTM data
    """
    mnistm = pkl.load(open('./data/mnistm_data.pkl', 'rb'))

    x_train = mnistm['x_train']
    x_test = mnistm['x_test']

    y_train = mnistm['y_train']
    y_test = mnistm['y_test']

    return x_train, y_train, x_test, y_test


def load_svhn():
    """
    Prepare SVHN data
    """
    # Load the data
    train_raw = loadmat('./data/svhndataset/Cropped digits/train_32x32.mat')
    test_raw = loadmat('./data/svhndataset/Cropped digits/test_32x32.mat')

    # Load images and labels
    x_train = np.array(train_raw['X'])
    x_test = np.array(test_raw['X'])

    y_train = train_raw['y']
    y_test = test_raw['y']

    # Fix the axes of the images
    x_train = np.moveaxis(x_train, -1, 0)
    x_test = np.moveaxis(x_test, -1, 0)

    y_train = fixLabel(y_train)
    y_test = fixLabel(y_test)

    return x_train, y_train, x_test, y_test


def load_usps():
    """
    Prepare USPS data
    """

    with h5py.File('./data/usps.h5', 'r') as hf:
        train = hf.get('train')
        x_train = train.get('data')[:]
        y_train = train.get('target')[:]
        test = hf.get('test')
        x_test = test.get('data')[:]
        y_test = test.get('target')[:]

    x_train = x_train.reshape([x_train.shape[0], 16, 16, 1])
    x_test = x_test.reshape([x_test.shape[0], 16, 16, 1])

    x_train = np.concatenate([x_train, x_train, x_train], 3)
    x_test = np.concatenate([x_test, x_test, x_test], 3)

    return x_train, y_train, x_test, y_test


def load_masks(handwritten):
    """
    Load masks from disk
    """
    if handwritten:
        path_to_data = "./data/mnist-masks/"
    else:
        path_to_data = "./data/digit-masks/"
    x_mask = []
    y_mask = []
    for i in range(10):
        name = path_to_data + 'image_' + str(i) + '.png'
        img = cv2.imread(name)

        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
        # cv2.imshow('Black white image', blackAndWhiteImage)
        # img = blackAndWhiteImage
        img = grayImage

        x_mask.append(img)
        y_mask.append(i)

    x_mask = np.array(x_mask)
    y_mask = np.array(y_mask)

    x_mask = np.expand_dims(x_mask, -1)
    # x_mask = np.concatenate([x_mask, x_mask, x_mask], 3)

    return x_mask, y_mask


def resize_images(x):
    """ 
    Resize images 
    """
    sz = (cfg.im_size, cfg.im_size)
    print('[INFO] Resizing images to {}...'.format(sz))
    x_resized = []
    for i in range(len(x)):
        img = x[i]
        resized_img = resize(img, sz) # , order=0)
        x_resized.append(resized_img)
    x = np.array(x_resized)
    return x


def fixLabel(labels):
    """ 
    Replace 10 in labels with 0
    """
    labels = labels.astype('int64')
    labels[labels == 10] = 0
    return labels


def create_all_pairs(x1, y1, x2, y2):
    """ 
    All positive and negative pairs between two arrays 
    """
    print('[INFO ] Creating pairs... ')
    pairs = []
    labels = []

    for i in range(len(x1)):
        for j in range(len(x2)):

            pairs += [[x1[i], x2[j]]]
            if y1[i] == y2[j]:
                labels += [1.]
            else:
                labels += [0.]

    return np.array(pairs).astype('float64'), np.array(labels).astype('float64')


def save_image_to_disk(name, im):
    """
    Save an image to the disk
    """
    if im.ndim == 2:
        im = im.reshape(im.shape[0], im.shape[1], 1)

    keras.preprocessing.image.save_img(name, im)


def save_1000mnist_to_disk():
    """
    Save 1000 mnist images to the disk
    """
    x_train, y_train, x_test, y_test = load_mnist()
    for i in range(1000):
        save_image_to_disk('./data/mnist100/{}_{}.png'.format(i,
                                                              y_train[i]), x_train[i])


def get_masks(r=0):
    x_mask, y_mask = load_masks(cfg.handwritten_masks)

    if not r == 0:
        x_mask = rotate_masks(x_mask , r)

    x_mask, y_mask = type_size_scale(x_mask, y_mask)

    return x_mask, y_mask


def unison_shuffled_copies(a, b):
    """ separate unison-shuffled arrays. """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def get_tiny_dataset(X, Y, max=10):
    """ get a subset of the orginal dataset with balanced classes. """
    print('[INFO] Orginal dataset size: {}'.format(X.shape[0]))
    tiny_X = []
    tiny_Y = []
    for i in range(10):
        j = 0
        for index, y in enumerate(Y):
            if y == i and j < max:
                # print(i, y, j)
                tiny_X.append(X[index])
                tiny_Y.append(y)
                j += 1
    tiny_X = np.asarray(tiny_X)
    tiny_Y = np.asarray(tiny_Y)

    tiny_X, tiny_Y = unison_shuffled_copies(tiny_X, tiny_Y)
    print('[INFO] New dataset size: {}'.format(tiny_X.shape[0]))


    return tiny_X, tiny_Y

