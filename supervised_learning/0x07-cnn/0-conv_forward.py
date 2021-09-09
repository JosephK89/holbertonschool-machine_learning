#!/usr/bin/env python3
"""Convolutional Forward Prop Module"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional neural network
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding is 'valid':
        ph = 0
        pw = 0
    elif padding is 'same':
        ph = ((((h_prev - 1) * sh) + kh - h_prev) // 2)
        pw = ((((w_prev - 1) * sw) + kw - w_prev) // 2)
    else:
        return
    images = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    'constant', constant_values=0)
    ch = ((h_prev + (2 * ph) - kh) // sh) + 1
    cw = ((w_prev + (2 * pw) - kw) // sw) + 1
    convoluted = np.zeros((m, ch, cw, c_new))
    for index in range(c_new):
        kernel_index = W[:, :, :, index]
        i = 0
        for h in range(0, (h_prev + (2 * ph) - kh + 1), sh):
            j = 0
            for w in range(0, (w_prev + (2 * pw) - kw + 1), sw):
                output = np.sum(
                    images[:, h:h + kh, w:w + kw, :] * kernel_index,
                    axis=1).sum(axis=1).sum(axis=1)
                output += b[0, 0, 0, index]
                convoluted[:, i, j, index] = activation(output)
                j += 1
            i += 1
    return convoluted
