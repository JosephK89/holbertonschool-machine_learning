#!/usr/bin/env python3
"""precision module"""
import numpy as np


def precision(confusion):
    """function that returns the precision"""
    return np.sum((confusion * np.identity(
        confusion.shape[0])) / np.sum(confusion, axis=0), axis=1)
