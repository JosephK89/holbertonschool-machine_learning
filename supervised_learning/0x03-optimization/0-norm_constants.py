#!/usr/bin/env python3
"""Normalization Module"""
import numpy as np


def normalization_constants(X):
    """this function returns the mean and the standard deviation"""
    mean = np.mean(X, axis=0)
    stdev = np.std(X, axis=0)
    return mean, stdev
