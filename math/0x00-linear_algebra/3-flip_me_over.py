#!/usr/bin/env python3
"""matrix_transpose"""


def transpose(matrix):
    """transpose function"""
    rows = len(matrix)
    columns = len(matrix[0])
    matrix_T = []
    for i in range(columns):
        row = []
        for j in range(rows):
           row.append(matrix[j][i])
        matrix_T.append(row)
    return matrix_T
