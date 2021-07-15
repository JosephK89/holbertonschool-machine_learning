#!/usr/bin/env python3
"""matrix multiplication"""


def mat_mul(mat1, mat2):
    """mat_mul function"""
    if len(mat1[0]) != len(mat2):
        return None
    matrix = [[sum(a * b for a, b in zip(mat1_rows, mat2_column)) for mat2_column in zip(*mat2)]for mat1_rows in mat1]
    return matrix
