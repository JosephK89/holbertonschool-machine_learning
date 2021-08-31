#!/usr/bin/env python3
"""build nn with keras"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """function that build nn with keras library"""
    sequential = []
    shape = (nx,)
    reg_l2 = K.regularizers.l2(lambtha)
    for i in range(len(layers)):
        if i is 0:
            sequential.append(K.layers.Dense(layers[i],
                                             activation=activations[i],
                                             kernel_regularizer=reg_l2,
                                             input_shape=shape))
        else:
            sequential.append(K.layers.Dropout(1 - keep_prob))
            sequential.append(K.layers.Dense(layers[i],
                                             activation=activations[i],
                                             kernel_regularizer=reg_l2))
    model = K.Sequential(sequential)
    return model
