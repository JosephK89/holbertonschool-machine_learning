#!/usr/bin/env python3
"""add_matrices2D"""


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
