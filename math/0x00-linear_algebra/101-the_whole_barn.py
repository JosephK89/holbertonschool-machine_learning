#!/usr/bin/env python3
"""add_matrices"""


def matrix_shape(matrix):
    """matrix_shape function"""
    matrix_S = [len(matrix)]
    while type(matrix[0]) != int:
        matrix_S.append(len(matrix[0]))
        matrix = matrix[0]
    return matrix_S


def add_arrays(arr1, arr2):
    """add_arrays function"""
    if len(arr1) != len(arr2):
        return None
    addition_arrays = []
    for i in range(len(arr1)):
        addition_arrays.append(arr1[i] + arr2[i])
    return addition_arrays


def add_matrices2D(mat1, mat2):
    """add_matrix2D function"""
    if len(mat1[0]) != len(mat2[0]):
        return None
    addition_matrices = []
    for row in range(len(mat1)):
        arr = []
        for column in range(len(mat1[0])):
            arr.append(mat1[row][column] + mat2[row][column])
        addition_matrices.append(arr)
    return addition_matrices


def add_matrices(mat1, mat2):
    """Matrices Addition """
    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)
    if shape1 != shape2:
        return None
    if type(mat1[0]) != list:
        return add_arrays(mat1, mat2)
    if len(shape1) == 2:
        return add_matrices2D(mat1, mat2)
    added_matrix = []
    for i in range(len(mat1)):
        added_matrix.append(add_matrices(mat1[i], mat2[i]))
    return added_matrix
