#!/usr/bin/env python3
"""add_arrays"""


def add_arrays(arr1, arr2):
    """add_arrays function"""
    if len(arr1) != len(arr2):
        return None
    addition_matrix = []
    for i in range(len(arr1)):
        addition_matrix.append(arr1[i] + arr2[i])
    return addition_matrix
