#!/usr/bin/env python3
"""test nn module"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """function that tests a nn"""
    return network.evaluate(data, labels, verbose=verbose)
