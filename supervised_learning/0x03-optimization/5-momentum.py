#!/usr/bin/env python3
"""Momentum Module"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """this function returns the variable and gradient"""
    Vdw = beta1 * v + (1 - beta1) * grad
    W = var - alpha * Vdw
    return W, Vdw
