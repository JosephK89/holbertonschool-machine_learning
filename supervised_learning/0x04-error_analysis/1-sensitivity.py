#!/usr/bin/env python3
"""sensitivity module"""
import numpy as np


def sensitivity(confusion):
    """function that return the sensitivity"""
    return np.sum((confusion * np.identity(
        confusion.shape[0])) / np.sum(confusion, axis=1), axis=1)
