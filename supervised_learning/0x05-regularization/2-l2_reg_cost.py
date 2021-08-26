#!/usr/bin/env python3
"""cost calculation module"""
import tensorflow as tf


def l2_reg_cost(cost):
    """function that calculates the cost"""
    regularization_losses = tf.losses.get_regularization_losses()
    return cost + regularization_losses
