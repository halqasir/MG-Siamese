#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 12:50:35 2021

@author: hiba_alqasir
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.manifold import TSNE
from config import cfg


def imshow_grid(images, title, shape=[2, 5]):
    """
    Plot images in a grid of a given shape.
    """
    fig = plt.figure(1)

    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

    size = shape[0] * shape[1]
    for i in range(size):
        grid[i].axis('off')
        grid[i].imshow(images[i], cmap='gray')  # The AxesGrid object work as a list of axes.

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
        fig.savefig('{}/examples_{}.png'.format(cfg.MODEL_DIR, title), dpi=fig.dpi)

    plt.show()
    plt.close(fig)


def plot_embedding_multi_domains(X, y, title, d=None, num_test=0):
    """
    Plot an embedding X with the class label y colored by the domain d.
    """
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    plt.ylim(-0.1, 1.1)
    plt.xlim(-0.1, 1.1)

    if d is not None:
        custom_lines = [
            Line2D([0], [0], color=plt.cm.PiYG(d[0] / 1.), lw=4),
            Line2D([0], [0], color=plt.cm.PiYG(d[-1] / 1.), lw=4)
        ]

        ax.legend(custom_lines, ['Target', 'Source'])
    else:
        d = np.tile([1., 0.], [num_test, 1])

    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.PiYG(d[i] / 1.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])

    plt.title(title)
    fig.savefig('{}/TSNE/tsne_{}.pdf'.format(cfg.MODEL_DIR, title), dpi=fig.dpi)

    plt.show()
    plt.close(fig)


def plot_embedding(X, y, title):
    """
    Plot an embedding X colored by class label y.
    """
    _classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

    fig = plt.figure(figsize=(10, 10))
    for i in range(10):
        inds = np.where(y == i)[0]
        plt.scatter(X[inds, 0], X[inds, 1], alpha=0.5, color=colors[i])
    plt.legend(_classes)
    fig.savefig('{}/TSNE/tsne_{}.pdf'.format(cfg.MODEL_DIR, title), dpi=fig.dpi)
    plt.show()
    plt.close(fig)


def plot_tsne_one_domain(emb, test_labels, num_test, title):
    """
    Compute tSne.
    """
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    emb_tsne = tsne.fit_transform(emb)
    plot_embedding_multi_domains(emb_tsne, test_labels.argmax(1), title, None, num_test)
    plot_embedding(emb_tsne, test_labels.argmax(1), title)


def plot_tsne(emb, test_labels, title, test_domain=None, num_test=0):
    """
    Compute tSne.
    """
    if not os.path.exists('{}/TSNE'.format(cfg.MODEL_DIR)):
        os.mkdir('{}/TSNE'.format(cfg.MODEL_DIR))

    # https://github.com/pumpikano/tf-dann/blob/master/MNIST-DANN.ipynb
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    emb_tsne = tsne.fit_transform(emb)

    plot_embedding(emb_tsne[:num_test], test_labels[:num_test].argmax(1), '{}_target'.format(title))
    plot_embedding(emb_tsne[num_test:], test_labels[num_test:].argmax(1), '{}_source'.format(title))

    plot_embedding_multi_domains(emb_tsne, test_labels.argmax(1), title, test_domain.argmax(1), num_test)
