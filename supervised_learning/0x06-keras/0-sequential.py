#!/usr/bin/env python3
"""build nn with keras"""
import tensorflow.keras as k


def build_model(nx, layers, activations, lambtha, keep_prob):
    """function that builds nn with keras library"""
    weights = K.initializers.VarianceScaling(mode="fan_avg")
    reg = K.regularizers.l2(lambtha)
    model = K.Sequential()
    model.add(K.layers.Dense(layers[0],
                             activation=activations[0],
                             kernel_initializer=weights,
                             kernel_regularizer=reg,
                             input_shape=(nx,)))
    for i in range(1, len(layers)):
        reg = K.regularizers.l2(lambtha)
        model.add(K.layers.Dropout(rate=(1 - keep_prob)))
        model.add(
            K.layers.Dense(layers[i],
                           activation=activations[i],
                           kernel_initializer=weights,
                           kernel_regularizer=reg)
        )
    return model 
