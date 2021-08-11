#!/usr/bin/env python3
"""forward propagation module"""
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """forward propagation function"""
    A = x
    for i in range(len(layer_sizes)):
        A = create_layer(A, layer_sizes[i], activations[i])
    return A
