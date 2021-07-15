#!/usr/bin/env python3
"""np_cat"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """nb_cat function"""
    return(np.concatenate((mat1, mat2), axis=axis))
