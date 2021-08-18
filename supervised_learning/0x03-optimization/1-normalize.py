#!/usr/bin/env python3
"""Normalization Module"""
import numpy as np


def normalize(X, m, s):
    """this function normalize a matrix"""
    X = (X - m) / s
    return X
