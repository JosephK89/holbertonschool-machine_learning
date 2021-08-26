#!/usr/bin/env python3
"""f1 score calculation module"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """function that returns the f1 score"""
    return 2 * (sensitivity(confusion) * precision(confusion)) / (
        sensitivity(confusion) + precision(confusion))
