#!/usr/bin/env python3
"""Batch Normalization Module"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """this function returns an activated tensor output"""
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=None,
                            kernel_initializer=w, name="layer")
    y = layer(prev)
    mean, variance = tf.nn.moments(y, [0])
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True)
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True)
    y_norm = tf.nn.batch_normalization(y, mean, variance, offset=beta,
                                       scale=gamma, variance_epsilon=1e-8)
    return (activation(y_norm))
