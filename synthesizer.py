#! -*- coding: utf-8 -*-

import tensorflow as tf
from position_embedding import Position_Embedding
from Feed_Forward import feed_forward
from Multi_Head_Attention import multi_head_attention
from Factorized_Dense_Synthesizer import Factorized_Dense_Synthesizer
from Factorized_Random_Synthesizer import Factorized_Random_Synthesizer
from Dense_Synthesizer import Dense_Synthesizer
from Random_Synthesizer import Random_Synthesizer
from Normlizition import layer_norm

'''
前向传播部分，输入是（batch_size, seq_len, word_size）形状
经过五种注意力结构的组合
输出是（batch_size, seq_len, word_size）形状
'''


def encoder(
        name,
        inputs,
        embedding_size,
        nb_layers,
        nb_head,
        a,b,
        size_per_head,
        initializer=None,
        V_len=None,
        training=True,
        keep_rate=None,
        activation='relu'):
    with tf.variable_scope(name):
        position = Position_Embedding(inputs=inputs, position_size=embedding_size)
        batch = tf.add(position,inputs)
        for i in range(nb_layers):
            mha_layer = multi_head_attention(
                                            Q=batch,
                                            K=batch,
                                            V=batch,
                                            nb_head=nb_head,
                                            size_per_head=size_per_head,
                                            initialzer=initializer,
                                            keep_rate=keep_rate,
                                            is_trainning=training,
                                            activation=activation,
                                            Q_len=V_len,
                                            V_len=V_len)
            ds_layer = Dense_Synthesizer(R=batch,
                                         V=batch,
                                         nb_head=nb_head,
                                         size_per_head=size_per_head,
                                         initialzer=initializer,
                                         keep_rate=keep_rate,
                                         is_trainning=training,
                                         activation=activation,
                                         X_len=V_len,
                                         V_len=V_len)
            fds_layer = Factorized_Dense_Synthesizer(R=batch,
                                                     V=batch,
                                                     a=a,
                                                     b=b,
                                                     nb_head=nb_head,
                                                     size_per_head=size_per_head,
                                                     initializer=initializer,
                                                     keep_rate=keep_rate,
                                                     is_trainning=training,
                                                     activation=activation,
                                                     X_len=V_len,
                                                     V_len=V_len,)
            rs_layer = Random_Synthesizer(name=name+'_'+str(i),
                                          V=batch,
                                          nb_head=nb_head,
                                          size_per_head=size_per_head,
                                          initializer=initializer,
                                          keep_rate=keep_rate,
                                          is_trainning=training,
                                          activation=activation,
                                          X_len=V_len,
                                          V_len=V_len)
            frs_layer = Factorized_Random_Synthesizer(name=name+'_'+str(i),
                                                      V=batch,
                                                      nb_head=nb_head,
                                                      size_per_head=size_per_head,
                                                      a=a,
                                                      b=b,
                                                      initializer=initializer,
                                                      keep_rate=keep_rate,
                                                      is_trainning=training,
                                                      activation=activation,
                                                      X_len=V_len,
                                                      V_len=V_len)
            mha_layer = tf.expand_dims(mha_layer,axis=2)
            ds_layer = tf.expand_dims(ds_layer,axis=2)
            fds_layer = tf.expand_dims(fds_layer,axis=2)
            rs_layer = tf.expand_dims(rs_layer,axis=2)
            frs_layer = tf.expand_dims(frs_layer,axis=2)
            multi_layer = tf.concat([mha_layer,ds_layer,fds_layer,rs_layer,frs_layer],axis=2)
            alpha = tf.get_variable(name=name+'_alpha_'+str(i),shape=[1,1,5,1],dtype=tf.float32,trainable=True)
            alpha = tf.nn.softmax(alpha,axis=2)
            multi_layer = tf.multiply(alpha,multi_layer)
            multi_layer = tf.reduce_mean(multi_layer,axis=2)
            add_layer_1 = tf.add(batch, multi_layer)
            ln_layer_1 = layer_norm(x=add_layer_1, scope=name + '_lnlayer_1_' + str(i))
            ff_layer = feed_forward(inputs=ln_layer_1,
                                    initializer=initializer,
                                    keep_rate=keep_rate,
                                    is_training=training,
                                    activition=activation)
            add_layer_2 = tf.add(ln_layer_1, ff_layer)
            batch = layer_norm(add_layer_2,scope=name +'_lnlayer_2_' +str(i))

        return batch


