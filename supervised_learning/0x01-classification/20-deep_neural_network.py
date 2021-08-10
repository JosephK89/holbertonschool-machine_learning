#!/usr/bin/env python3
"""Deep Neural Network Module"""
import numpy as np


class DeepNeuralNetwork:
    """DeepNeuralNetwork Class"""
    def __init__(self, nx, layers):
        """Data initialization"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) != list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        layer = 1
        layer_size = nx
        for i in layers:
            if type(i) != int or i <= 0:
                raise TypeError("layers must be a list of positive integers")
            w = "W" + str(layer)
            b = "b" + str(layer)
            self.weights[w] = np.random.randn(
                i, layer_size) * np.sqrt(2/layer_size)
            self.weights[b] = np.zeros((i, 1))
            layer += 1
            layer_size = i

    @property
    def L(self):
        """Getter function for L"""
        return self.__L

    @property
    def cache(self):
        """Getter function for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter function for weights"""
        return self.__weights

    def forward_prop(self, X):
        """forward propagation function"""
        self.__cache["A0"] = X
        for i in range(1, self.__L + 1):
            w = "W" + str(i)
            b = "b" + str(i)
            a = "A" + str(i - 1)
            Z = np.matmul(self.__weights[w],
                          self.__cache[a]) + self.__weights[b]
            a_new = "A" + str(i)
            self.__cache[a_new] = 1 / (1 + np.exp(-Z))
        Act = "A" + str(self.__L)
        return (self.__cache[Act], self.__cache)

    def cost(self, Y, A):
        """model cost function"""
        cost_array = np.multiply(np.log(A), Y) + np.multiply((
            1 - Y), np.log(1.0000001 - A))
        cost = -np.sum(cost_array) / len(A[0])
        return cost

    def evaluate(self, X, Y):
        """evaluate neurons function"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        return (np.where(A > 0.5, 1, 0), cost)
