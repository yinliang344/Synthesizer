import tensorflow as tf
from Mask import Mask
from Dense import Dense

'''
Multi-Head Attention的实现
'''
def multi_head_attention(
        Q,
        K,
        V,
        nb_head,
        size_per_head,
        initialzer=None,
        keep_rate=None,
        is_trainning=None,
        activation='relu',
        Q_len=None,
        V_len=None):
    # 对Q、K、V分别作线性映射
    query = Dense(inputs=Q,
                  output_size=nb_head * size_per_head,
                  keep_rate=keep_rate,
                  is_trainning=is_trainning,
                  initializer=initialzer,
                  activition=activation,
                  bias=False)
    query = tf.reshape(query, (-1, tf.shape(query)[1], nb_head, size_per_head))
    query = tf.transpose(query, [0, 2, 1, 3])
    key = Dense(inputs=K,
                output_size=nb_head * size_per_head,
                keep_rate=keep_rate,
                is_trainning=is_trainning,
                initializer=initialzer,
                activition=activation,
                bias=False)
    key = tf.reshape(key, (-1, tf.shape(key)[1], nb_head, size_per_head))
    key = tf.transpose(key, [0, 2, 1, 3])
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
    A = tf.matmul(query, key, transpose_b=True) / tf.sqrt(float(size_per_head))
    A = tf.transpose(A, [0, 3, 2, 1])
    A = Mask(A, V_len, mode='add')
    A = tf.transpose(A, [0, 3, 2, 1])
    A = tf.nn.softmax(A)
    # 输出并mask
    output = tf.matmul(A, value)
    output = tf.transpose(output, [0, 2, 1, 3])
    output = tf.reshape(
        output, (-1, tf.shape(output)[1], nb_head * size_per_head))
    output = Mask(output, Q_len, 'mul')
    return output