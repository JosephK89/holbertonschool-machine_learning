#!/usr/bin/env python3
"""Adam Optimization Module"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """this function returns an updated variable,
    new fist moment, new second moment"""
    Vdw = beta1 * v + (1 - beta1) * grad
    Sdw = beta2 * s + (1 - beta2) * (grad**2)
    Vdwc = Vdw / (1 - beta1**t)
    Sdwc = Sdw / (1 - beta2**t)
    W = var - alpha * Vdwc / (Sdwc ** (1/2) + epsilon)
    return W, Vdw, Sdw
