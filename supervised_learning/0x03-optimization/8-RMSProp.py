#!/usr/bin/env python3
"""RMSProp Module"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """this function returns a tensor with RMSProp"""
    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                          decay=beta2, epsilon=epsilon)
    train = optimizer.minimize(loss)
    return train
