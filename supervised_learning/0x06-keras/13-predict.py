  #!/usr/bin/env python3
"""predict a nn module"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """function that predicts a nn"""
    return network.predict(data, verbose=verbose)
