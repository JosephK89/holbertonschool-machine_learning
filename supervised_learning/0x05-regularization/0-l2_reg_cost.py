#!/usr/bin/env python3
"""l2 regularization module"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """function that return the cost using l2 regularization"""
    w = [weights["W" + str(i+1)] for i in range (L)]
    w = [np.linalg.norm(w) ** 2 for w in w]
    cost = cost + (lambtha * sum(w)) / (2 * m)
    return cost
