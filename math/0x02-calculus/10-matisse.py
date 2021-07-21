#!/usr/bin/env python3
"""derivative of a polynomial"""


def poly_derivative(poly):
    """poly_derivative function"""
    if type(poly) != list or len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]
    if poly is None:
        return None
    nlist = list(range(len(poly)-1))
    for i in range(len(nlist)):
        if type(i) != int:
            return None
        nlist[i] = poly[i + 1] * (i + 1)
    return nlist
