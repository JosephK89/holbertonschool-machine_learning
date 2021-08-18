#!/usr/bin/env python3
"""Adam Optimization Module"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """this function returns a tensor Adam optimized"""
    optimizer = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    train = optimizer.minimize(loss)
    return train
