#!/usr/bin/env python3
"""Pooling Forward Prop Module"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over pooling layer of a neural network
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    ph = ((h_prev - kh) // sh) + 1
    pw = ((w_prev - kw) // sw) + 1
    pooled = np.zeros((m, ph, pw, c_prev))
    i = 0
    for h in range(0, (h_prev - kh + 1), sh):
        j = 0
        for w in range(0, (w_prev - kw + 1), sw):
            if mode == 'max':
                output = np.max(A_prev[:, h:h + kh, w:w + kw, :],
                                axis=(1, 2))
            elif mode == 'avg':
                output = np.average(A_prev[:, h:h + kh, w:w + kw, :],
                                    axis=(1, 2))
            else:
                pass
            pooled[:, i, j, :] = output
            j += 1
        i += 1
    return pooled
