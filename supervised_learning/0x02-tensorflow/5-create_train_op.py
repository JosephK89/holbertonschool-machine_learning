#!/usr/bin/env python3
"""training operation module"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """training operation function"""
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train = optimizer.minimize(loss)
    return train
