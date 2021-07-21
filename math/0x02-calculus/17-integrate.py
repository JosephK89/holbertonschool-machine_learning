#!/usr/bin/env python3
"""integral of a polynomial"""


def poly_integral(poly, C=0):
    """poly_integral function"""
    nlist = []
    i = 0

    if type(poly) != list or len(poly) == 0 or type(C) != int:
        return None
    nlist.append(C)
    if sum(poly) == 0:
        return new_L
    if len(poly) == 1:
        nlist.append(poly[0])
        return nlist
    while i < len(poly):
        if poly[i] % (i + 1) == 0:
            nlist.append(int(poly[i]/(i + 1)))
        else:
            nlist.append(poly[i]/(i + 1))
        i += 1
    return nlist
