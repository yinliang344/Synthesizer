#! -*- coding: utf-8 -*-

import tensorflow as tf

'''
普通的全连接
inputs是一个二阶或二阶以上的张量，即形如(batch_size,...,input_size)。
只对最后一个维度做矩阵乘法，即输出一个形如(batch_size,...,ouput_size)的张量。
'''


def Dense(
        inputs,
        output_size,
        initializer=None,
        keep_rate=None,
        is_trainning=True,
        activition=None,
        bias=False):

    outputs = tf.layers.dense(
        inputs=inputs,
        units=output_size,
        use_bias=bias,
        kernel_initializer=initializer)
    # outputs = tf.layers.batch_normalization(outputs,training=is_trainning)
    if activition is 'relu':
        outputs = tf.nn.relu(outputs)
    elif activition is 'leaky_relu':
        outputs = tf.nn.leaky_relu(outputs)
    elif activition is 'sigmoid':
        outputs = tf.nn.sigmoid(outputs)
    if keep_rate is not None:
        outputs = tf.nn.dropout(outputs, keep_prob=keep_rate)
    return outputs