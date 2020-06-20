#! -*- coding: utf-8 -*-

import tensorflow as tf

def conv2D(
        inputs,
        kernel_shape,
        strides,
        padding,
        kernel_name,
        training,
        activation='relu',
        dropuot_rate=None):
    kernel = tf.get_variable(
        dtype=tf.float32,
        shape=kernel_shape,
        name=kernel_name,
        regularizer=tf.contrib.layers.l2_regularizer(10e-6),
        initializer=tf.contrib.layers.xavier_initializer())
    conv_output = tf.nn.conv2d(
        input=inputs,
        filter=kernel,
        strides=strides,
        padding=padding)
    conv_output = tf.layers.batch_normalization(
        inputs=conv_output, training=training)
    if activation is 'relu':
        conv_output = tf.nn.relu(conv_output)
    elif activation is 'leaky_relu':
        conv_output = tf.nn.leaky_relu(conv_output)
    if dropuot_rate is not None:
        conv_output = tf.nn.dropout(conv_output, keep_prob=dropuot_rate)
    return conv_output