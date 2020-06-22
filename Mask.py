#! -*- coding: utf-8 -*-

import tensorflow as tf


'''
inputs是一个二阶以上的张量，代表输入序列，比如形如(batch_size, seq_len, input_size)的张量；
seq_len是一个形如(batch_size,)的张量，代表每个序列的实际长度，多出部分都被忽略；
mode分为mul和add，mul是指把多出部分全部置零，一般用于全连接层之前；
add是指把多出部分全部减去一个大的常数，一般用于softmax之前。
'''

def Mask(inputs, seq_true_len=None,truncature_len=None, mode='mul'):
    if seq_true_len is None:
        return inputs
    else:
        mask = tf.cast(tf.sequence_mask(seq_true_len, truncature_len), tf.float32)
        for _ in range(len(inputs.shape) - 2):
            mask = tf.expand_dims(mask, 2)
        if mode == 'mul':
            return inputs * mask
        if mode == 'add':
            return inputs - (1 - mask) * 1e12