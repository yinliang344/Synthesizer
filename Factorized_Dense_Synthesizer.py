#! -*- coding: utf-8 -*-

import tensorflow as tf
from Mask import Mask
from Dense import Dense

'''
Factorized Dense Synthesizer的实现
'''

def Factorized_Dense_Synthesizer(X, V, nb_head,a,b,
                      size_per_head,initialzer=None,
                      keep_rate=None,is_trainning=None,
                      activation='relu',X_len=None,V_len=None):

    B_1 = Dense(inputs=X,
                  output_size=b,
                  keep_rate=keep_rate,
                  is_trainning=is_trainning,
                  initializer=initialzer,
                  activition=activation,
                  bias=True)
    B_1 = Dense(inputs=B_1,
              output_size=b,
              keep_rate=keep_rate,
              is_trainning=is_trainning,
              initializer=initialzer,
              activition=None,
              bias=True)

    A_1 = Dense(inputs=X,
                output_size=a,
                keep_rate=keep_rate,
                is_trainning=is_trainning,
                initializer=initialzer,
                activition=activation,
                bias=True)
    A_1 = Dense(inputs=A_1,
                output_size=a,
                keep_rate=keep_rate,
                is_trainning=is_trainning,
                initializer=initialzer,
                activition=None,
                bias=True)
    B_1 = tf.tile(B_1,[1,1,a])
    A_1 = tf.tile(A_1,[1,1,b])
    AB = tf.multiply(A_1,B_1)
    X = tf.reshape(AB, (-1, tf.shape(X)[1], nb_head, size_per_head))
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
    B = X / tf.sqrt(float(size_per_head))
    B = tf.transpose(B, [0, 3, 2, 1])
    B = Mask(B, V_len, 'add')
    B = tf.transpose(B, [0, 3, 2, 1])
    B = tf.nn.softmax(B)
    # 输出并mask
    output = tf.matmul(B, value)
    output = tf.transpose(output, [0, 2, 1, 3])
    output = tf.reshape(output, (-1, tf.shape(output)[1], nb_head * size_per_head))
    output = Mask(output, X_len, 'mul')
    return output