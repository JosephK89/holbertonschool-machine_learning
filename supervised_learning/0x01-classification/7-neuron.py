#!/usr/bin/env python3
"""neuron class module"""
import numpy as np
import matplotlib.pyplot as plt


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

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """gradient descent function"""
        cost = self.cost(Y, A)
        dz = A - Y
        dw = (1 / len(Y[0])) * np.matmul(dz, X.T)
        db = (1 / len(Y[0])) * np.sum(dz)
        self.__W = self.__W - alpha * dw
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """train function"""
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
        return self.evaluate(X, Y)
        costs = []
        iters = []
        for i in range(iterations + 1):
            A, cost = self.evaluate(X, Y)
            if i != iterations:
                self.forward_prop(X)
                self.gradient_descent(X, Y, self.__A, alpha)
            if ((i % step == 0 or i == 0 or i == iterations) and
                verbose is True):
                print("Cost after {} iterations: {}".format(i, cost))
                costs.append(cost)
                iters.append(i)
        if graph is True:
            plt.plot(iters, costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.xlim(0, iterations)
            plt.show()
        return (A, cost)
