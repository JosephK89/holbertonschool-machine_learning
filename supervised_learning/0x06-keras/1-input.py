#!/usr/bin/env python3
"""build nn with keras"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """function that builds nn model with keras library"""
    weights = K.initializers.VarianceScaling(mode="fan_avg")
    reg = K.regularizers.l2(lambtha)
    inputs = K.Input(shape=(nx,))
    layer = K.layers.Dense(layers[0],
                           activation=activations[0],
                           kernel_initializer=weights,
                           kernel_regularizer=reg)(inputs)
    for i in range(1, len(layers)):
        drop = K.layers.Dropout(rate=(1 - keep_prob))(layer)
        reg = K.regularizers.l2(lambtha)
        layer = K.layers.Dense(layers[i],
                               activation=activations[i],
                               kernel_initializer=weights,
                               kernel_regularizer=reg)(drop)
    return K.Model(inputs=inputs, outputs=layer)
