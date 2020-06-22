#! -*- coding: utf-8 -*-

import tensorflow as tf
from Mask import Mask
from Dense import Dense

'''
Factorized Dense Synthesizer的实现
'''

def Factorized_Dense_Synthesizer(R, V, nb_head,a,b,
                      size_per_head,initializer=None,
                      keep_rate=None,is_trainning=None,
                      activation='relu',X_len=None,V_len=None):

    B = Dense(inputs=R,
                  output_size=b,
                  keep_rate=keep_rate,
                  is_trainning=is_trainning,
                  initializer=initializer,
                  activition=activation,
                  bias=True)
    B = Dense(inputs=B,
              output_size=b*nb_head,
              keep_rate=keep_rate,
              is_trainning=is_trainning,
              initializer=initializer,
              activition=None,
              bias=True)

    A = Dense(inputs=R,
                output_size=a,
                keep_rate=keep_rate,
                is_trainning=is_trainning,
                initializer=initializer,
                activition=activation,
                bias=True)
    A = Dense(inputs=A,
                output_size=a*nb_head,
                keep_rate=keep_rate,
                is_trainning=is_trainning,
                initializer=initializer,
                activition=None,
                bias=True)
    B = tf.tile(B,[1,1,a])
    A = tf.tile(A,[1,1,b])
    AB = tf.multiply(A,B)
    X = tf.reshape(AB, (-1, tf.shape(AB)[1], nb_head, a*b))
    X = tf.transpose(X, [0, 2, 1, 3])

    value = Dense(inputs=V,
                  output_size=nb_head * size_per_head,
                  keep_rate=keep_rate,
                  is_trainning=is_trainning,
                  initializer=initializer,
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