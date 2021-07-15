#!/usr/bin/env python3
"""matrix_shape"""


def matrix_shape(matrix):
    """matrix_shape function"""
    matrix_S = [len(matrix)]
    while type(matrix[0]) != int:
        matrix_S.append(len(matrix[0]))
        matrix = matrix[0]
    return matrix_S
