#!/usr/bin/env python3
"""save and load a model's configuration in JSON format"""
import tensorflow.keras as K


def save_config(network, filename):
    """function that saves model's configuration in JSON format"""
    with open(filename, "w") as json_file:
        json_file.write(network.to_json())


def load_config(filename):
    """function that loads model's configuration in JSON format"""
    with open(filename, 'r') as f:
        return K.models.model_from_json(f.read())
