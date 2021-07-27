#!/usr/bin/env python3
"""poisson distribution module"""


class Poisson:
    """"Poisson class"""
    def __init__(self, data=None, lambtha=1.):
    """initialization"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """pmf function"""
        if k < 0:
            return 0
        if type(k) is not int:
            k = int(k)
        k_factorial = 1
        if k != 0:
            for i in range(2, k+1):
                k_factorial = k_factorial * i
        return ((self.lambtha ** (k)) *
                (2.7182818285 ** (-(self.lambtha)))) / k_factorial

    def cdf(self, k):
        """cdf function"""
        if k < 0:
            return 0
        if type(k) is not int:
            k = int(k)
        cdf = 0
        for i in range(0, k+1):
            cdf = cdf + self.pmf(i)
        return cdf
