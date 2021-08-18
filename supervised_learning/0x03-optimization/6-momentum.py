#!/usr/bin/env python3
"""Momentum Module"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """this function create a momentum operation"""
    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    train = optimizer.minimize(loss)
    return train
