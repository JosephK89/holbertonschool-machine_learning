#!/usr/bin/env python3
"""Pooling Back Prop Module"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs backward propagation over a pooling layer of neural network
    """
    m, h_new, w_new, c = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    dA_prev = np.zeros((m, h_prev, w_prev, c))
    for ex in range(m):
        for kernel_index in range(c):
            for h in range(h_new):
                for w in range(w_new):
                    i = h * sh
                    j = w * sw
                    if mode is 'max':
                        pool = A_prev[ex, i: i + kh, j: j + kw, kernel_index]
                        mask = np.where(pool == np.max(pool), 1, 0)
                    elif mode is 'avg':
                        mask = np.ones((kh, kw))
                        mask /= (kh * kw)
                    dA_prev[ex, i: i + kh, j: j + kw, kernel_index] += (
                        mask * dA[ex, h, w, kernel_index])
    return dA_prev
