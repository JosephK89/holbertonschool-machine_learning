#!/usr/bin/env python3
"""Functions concatenating two matrices along an axis"""


def cat_matrices(mat1, mat2, axis=0):
    """Function that concatenates two matrices along a specific axis"""
    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)
    del shape1[axis]
    del shape2[axis]
    return None if shape1 != shape2 else cat_matrices_comp(mat1, mat2, axis)


def cat_matrices_comp(mat1, mat2, axis):
    """Recursive function concatenating two matrices along a specific axis"""
    if axis != 0:
        return [cat_matrices_comp(u, v, axis - 1) for u, v in zip(mat1, mat2)]
    return matrix_copy(mat1 + mat2)


def matrix_copy(matrix):
    """Function that returns a deep copy of the matrix"""
    if len(matrix) != 0 and isinstance(matrix[0], list):
        return list(map(matrix_copy, matrix))
    return matrix[:]


def matrix_shape(matrix):
    """matrix_shape function"""
    matrix_S = [len(matrix)]
    while type(matrix[0]) != int:
        matrix_S.append(len(matrix[0]))
        matrix = matrix[0]
    return matrix_S
