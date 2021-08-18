#!/usr/bin/env python3
"""Output Normalization Module"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """"this function returns a normalized output"""
    u = Z.mean(axis=0)
    s2 = Z.var(axis=0)
    Znorm = (Z - u) / (s2 + epsilon) ** (1/2)
    Zn = gamma * Znorm + beta
    return Zn
