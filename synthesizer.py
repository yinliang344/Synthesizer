#! -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from position_embedding import Position_Embedding
from Feed_Forward import feed_forward
from Multi_Head_Attention import multi_head_attention

'''
前向传播encoder部分，输入是（batch_size*2, seq_len, word_size）形状，经过multi_head_attention
然后残差连接和norm，再经过两层全连接，最后残差连接和norm
输出是（batch_size, seq_len, word_size）形状
'''


def encoder(
        name,
        inputs,
        embedding_size,
        nb_layers,
        nb_head,
        size_per_head,
        initializer=None,
        Q_len=None,
        V_len=None,
        training=True,
        keep_rate=None,
        activition='relu'):
    with tf.variable_scope(name):
        position = Position_Embedding(
            inputs=inputs, position_size=embedding_size)
        batch = tf.concat([position, inputs], axis=-1)
        for i in range(nb_layers):
            mha_layer = multi_head_attention(
                Q=batch,
                K=batch,
                V=batch,
                nb_head=np.shape(batch)[2] //size_per_head,
                size_per_head=size_per_head,
                initialzer=initializer,
                keep_rate=keep_rate,
                is_trainning=training,
                activation=activition,
                Q_len=Q_len,
                V_len=V_len)
            add_layer_1 = tf.add(batch, mha_layer)
            ln_layer_1 = layer_norm(
                x=add_layer_1, scope=name + '_lnlayer_1_' + str(i))
            ff_layer = feed_forward(inputs=ln_layer_1,
                                    initializer=initializer,
                                    keep_rate=keep_rate,
                                    is_training=training,
                                    activition=activition)
            add_layer_2 = tf.add(ln_layer_1, ff_layer)
            batch = layer_norm(add_layer_2,scope=name +'_lnlayer_2_' +str(i))

        return batch


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


def layer_norm(x, scope='layer_norm'):
    '''
    :param x: the tensor with shape (batch_size,sq_len,hidden_size) or (batch_size,hidden_size)
    :param scope: the name of layer
    :return:
    '''
    return tf.contrib.layers.layer_norm(
        x, center=True, scale=True, scope=scope)
