#! -*- coding: utf-8 -*-

import tensorflow as tf
from Mask import Mask
from Dense import Dense

'''
Random Synthesizer的实现
'''

def Random_Synthesizer(name,V, nb_head,
                      size_per_head,initialzer=None,
                      keep_rate=None,is_trainning=None,
                      activation='relu',X_len=None,V_len=None):

    AB = tf.get_variable(name=name,
                         shape=[1,tf.shape(V)[1],nb_head*size_per_head],
                         dtype=tf.float32,
                         initializer=initialzer,
                         trainable=True)
    X = tf.reshape(AB, (-1, tf.shape(AB)[1], nb_head, size_per_head))
    X = tf.transpose(X, [0, 2, 1, 3])

    value = Dense(inputs=V,
                  output_size=nb_head * size_per_head,
                  keep_rate=keep_rate,
                  is_trainning=is_trainning,
                  initializer=initialzer,
                  activition=activation,
                  bias=False)
    value = tf.reshape(value, (-1, tf.shape(value)[1], nb_head, size_per_head))
    value = tf.transpose(value, [0, 2, 1, 3])
    # 计算内积，然后mask，然后softmax
    X = X / tf.sqrt(float(size_per_head))
    X = tf.transpose(X, [0, 3, 2, 1])
    X = Mask(X, V_len, 'add')
    X = tf.transpose(X, [0, 3, 2, 1])
    X = tf.nn.softmax(X)
    # 输出并mask
    output = tf.matmul(X, value)
    output = tf.transpose(output, [0, 2, 1, 3])
    output = tf.reshape(output, (-1, tf.shape(output)[1], nb_head * size_per_head))
    output = Mask(output, X_len, 'mul')
    return output