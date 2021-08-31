#!/usr/bin/env python3
"""save and load a model's weights"""
import tensorflow.keras as K


def save_weights(network, filename, format='h5'):
    """function that saves model's weights"""
    network.save_weights(filename, format)


def load_weights(network, filename):
    """function that loads model's weights"""
    return network.load_weights(filename)
