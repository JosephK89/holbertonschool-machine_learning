#!/usr/bin/env python3
"""cat_matrices2D"""


def cat_matrices2D(mat1, mat2, axis=0):
    """cat_matrices2D function"""
    matrix = []
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        for i in mat1:
            matrix.append(x[:])
        for i in mat2:
            matrix.append(rows[:])
        return matrix
    if axis == 1 and len(mat1) == len(mat2):
        for i in range(len(mat1)):
            matrix.append(mat1[rows] + mat2[rows])
        return matrix
    return None
