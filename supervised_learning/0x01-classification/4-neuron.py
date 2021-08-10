#!/usr/bin/env python3
"""neuron class module"""
import numpy as np


class Neuron:
    """Neuron class"""
    def __init__(self, nx):
        """Class Initialization"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

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

    def forward_prop(self, X):
        """forward propagation function"""
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """model cost function"""
        cost_array = np.multiply(np.log(A), Y) + np.multiply((
            1 - Y), np.log(1.0000001 - A))
        cost = -np.sum(cost_array) / len(A[0])
        return cost

    def evaluate(self, X, Y):
        """evaluate neurons function"""
        self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        return (np.where(self.__A > 0.5, 1, 0), cost)
