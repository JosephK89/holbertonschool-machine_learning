#!/usr/bin/env python3
"""optimize model"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """optimize model function"""
    A = K.optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(loss='categorical_crossentropy', optimizer=A,
                    metrics=['accuracy'])
