#! -*- coding: utf-8 -*-

import tensorflow as tf

def layer_norm(x, scope='layer_norm'):
    '''
    :param x: the tensor with shape (batch_size,sq_len,hidden_size) or (batch_size,hidden_size)
    :param scope: the name of layer
    :return:
    '''
    ln = tf.contrib.layers.layer_norm(x, center=True, scale=True, scope=scope)
    return ln