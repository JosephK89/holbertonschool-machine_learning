#!/usr/bin/env python3
"""Shuffle Data Module"""
import numpy as np


def shuffle_data(X, Y):
    """this function returns shuffled X and Y matrices"""
    perm = X.shape[0]
    shuff_op = np.random.permutation(perm)
    return X[shuff_op], Y[shuff_op]
