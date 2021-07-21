#!/usr/bin/env python3
"""totalsum"""


def summation_i_squared(n):
    """summation_i_squared function"""
    if type(n) != int or n < 1:
        return None
    return (n * (n + 1) * (2 * n + 1)) // 6
