#! -*- coding: utf-8 -*-

from Dense import Dense

def feed_forward(
        inputs,
        initializer=None,
        keep_rate=None,
        is_training=True,
        activition='relu'):
    shapes = int(inputs.shape[-1])
    dense = Dense(inputs=inputs,
                  output_size=shapes * 2,
                  initializer=initializer,
                  keep_rate=keep_rate,
                  is_trainning=is_training,
                  activition=activition)
    dense = Dense(inputs=dense,
                  output_size=shapes,
                  initializer=initializer,
                  keep_rate=keep_rate,
                  is_trainning=is_training,
                  activition=activition)
    return dense