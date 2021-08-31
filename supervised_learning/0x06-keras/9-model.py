#!/usr/bin/env python3
"""save and load an entire model"""
import tensorflow.keras as K


def save_model(network, filename):
    """function that saves model"""
    network.save(filename)


def load_model(filename):
    """function that loads model"""
    return K.models.load_model(filename)
