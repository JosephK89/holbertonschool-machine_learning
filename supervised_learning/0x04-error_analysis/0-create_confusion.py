#!/usr/bin/env python3
"""confusion_matrix module"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """function that create a confusion matrix"""
    return np.matmul(labels.T, logits)    
