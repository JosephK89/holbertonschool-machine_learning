#!/usr/bin/env python3
"""Moving Average Module"""
import numpy as np


def moving_average(data, beta):
    """this function return the moving average"""
    vp = 0
    v_weighted = []
    for i in range(len(data)):
        v = (vp * beta + (1 - beta) * data[i])
        v_weighted.append(v / (1 - beta**(i + 1)))
        vp = v
    return v_weighted
