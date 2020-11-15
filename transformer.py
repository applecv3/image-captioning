import tensorflow.compat.v1 as tf
import numpy as np


def scaled_self_attention(query, key, value, masking):

    dim = tf.cast(tf.shape(query)[-1], tf.float32)
    attention_score = tf.matmul(query, tf.transpose(key, [0, 2, 1])) / tf.sqrt(dim)

    if masking:

        mask = tf.ones_like(attention_score)
        mask = tf.linalg.LinearOperatorLowerTriangular(mask).to_dense()
        negative_matrix = tf.ones_like(mask) * float('-inf')  # to get softmaxed values to be zero
        attention_score = tf.where(tf.equal(mask, 1), attention_score, negative_matrix)

    attention_map = tf.nn.softmax(attention_score)
    attention_map = tf.matmul(attention_map, value)

    return attention_map


def multi_head_attention(query, key, value, n_heads, masking=False):

    dim = query.shape[-1]

    query = tf.layers.dense(query, dim, activation=tf.nn.relu)
    key = tf.layers.dense(key, dim, activation=tf.nn.relu)
    value = tf.layers.dense(value, dim, activation=tf.nn.relu)

    query = tf.concat(tf.split(query, n_heads, axis=-1), axis=0)
    key = tf.concat(tf.split(key, n_heads, axis=-1), axis=0)
    value = tf.concat(tf.split(value, n_heads, axis=-1), axis=0)

    attention_map = scaled_self_attention(query, key, value, masking)
    attention_map = tf.concat(tf.split(attention_map, n_heads, axis=0), axis=-1)
    attention_map = tf.layers.dense(attention_map, dim, activation=tf.nn.relu)

    return attention_map


def norm(x, eps=1e-6):

    mean = tf.math.reduce_mean(x, -1, keepdims=True)
    std = tf.math.reduce_std(x, -1, keepdims=True)

    x = (x - mean) / (std + eps)

    return x


def add_and_norm(input1, input2, is_training, dropout):

    fused_feature = input1 + tf.layers.dropout(input2, rate=dropout, training=is_training)
    normed_feature = norm(fused_feature)

    return normed_feature


def ffn(x, ffn_dim):

    dim = x.shape[-1]

    x = tf.layers.dense(x, ffn_dim, activation=tf.nn.relu)
    x = tf.layers.dense(x, dim)

    return x


def decoder(x, encoded_image, ffn_dim, n_heads, dropout_rate, is_training):

    x = add_and_norm(x, multi_head_attention(x, x, x, n_heads, True), is_training, dropout_rate)
    x = add_and_norm(x, multi_head_attention(x, encoded_image, encoded_image, n_heads), is_training, dropout_rate)
    x = add_and_norm(x, ffn(x, ffn_dim), is_training, dropout_rate)

    return x


def position_encoding(dim, caption_length):

    position_matrix = np.array([[pos/np.power(10000, 2*(i//2)/dim) for i in range(dim)] for pos in range(caption_length)])

    position_matrix[:, ::2] = np.sin(position_matrix[:, ::2])
    position_matrix[:, 1::2] = np.cos(position_matrix[:, 1::2])

    return tf.constant(position_matrix, tf.float32)


def decoder_layer(x, encoded_image, params, is_training):

    encoded_image = tf.layers.dense(encoded_image, params.embedding_dim)

    for _ in range(params.n_layer):

        x = decoder(x, encoded_image, params.ffn_dim, params.n_heads, params.dropout_rate, is_training)

    return x
