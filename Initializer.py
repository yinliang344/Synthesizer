#! -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
def Xavier_initializer(node_in, node_out):
    '''
    :param node_in: the number of input size
    :param node_out: the number of output size
    :return: a weight matrix
    '''
    W = tf.div(tf.Variable(np.random.randn(node_in,node_out).astype('float32')),np.sqrt(node_in))
    return W


def He_initializer(node_in, node_out):
    '''
    :param node_in: the number of input size
    :param node_out: the number of output size
    :return: a weight matrix
    '''
    W = tf.div(tf.Variable(np.random.randn(node_in, node_out).astype('float32')), np.sqrt(node_in / 2))
    return W