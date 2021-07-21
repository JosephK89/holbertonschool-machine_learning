#!/usr/bin/env python3
"""totalsum"""


def summation_i_squared(n):
    """summation_i_squared function"""
    if type(n) != int or n < 1:
        return None
    sum = 0
    for i in range(1,n+1):
        sum += i**2
    return sum
