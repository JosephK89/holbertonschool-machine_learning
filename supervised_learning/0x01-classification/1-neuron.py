#!/usr/bin/env python3
"""neuron class module"""
import numpy as np


class Neuron:
    """Neuron class"""
    def __init__(self, nx):
        """Data Initialization"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0

    @property
    def W(self):
        """getter function for weights"""
        return self.__W

    @property
    def b(self):
        """getter function for biases"""
        return self.__b

    @property
    def A(self):
        """getter function for A"""
        return self.__A
