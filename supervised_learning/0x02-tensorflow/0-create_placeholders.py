#!/usr/bin/env python3
"""placeholder module"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """create_placeholder function"""
    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")
    return (x, y)
