#!/usr/bin/env python3
"""save and load a model's configuration in JSON format"""
import tensorflow.keras as K


def save_weights(network, filename, format='h5'):
    """function that saves model's configuration in JSON format"""
    network.save_weights(filename, format)


def load_weights(network, filename):
    """function that loads model's configuration in JSON format"""
    return network.load_weights(filename)

