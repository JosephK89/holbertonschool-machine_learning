#!/usr/bin/env python3
"""one-hot encode module"""
import numpy as np


def one_hot_encode(Y, classes):
    """this function will convert a numeric label vector to one hot matrix"""
    if type(Y) is not np.ndarray or len(Y) == 0:
        return None
    if type(classes) is not int or classes <= np.max(Y):
        return None
    A = np.eye(classes)[Y]
    return A.T
