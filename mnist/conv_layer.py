import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

def conv_layer(input_tensor, name, kernel_size, n_output_channels,
               padding_mode='SAME', strides=(1, 1, 1, 1)):
    with tf.variable_scope(name):
        # 入力チャンネルの個数(n_input_channels)を取得:
        #  input_tensornの形状: {バッチ x 幅 x 高さ x チャネル数}
        input_shape = input_tensor.get_shape().as_list()
        n_input_channels = input_shape[-1]

        weights_shape = (list(kernel_size) +
                         [n_input_channels, n_output_channels])

        weights = tf.get_variable(name='_weights', shape=weights_shape)
        print(weights)
        biases = tf.get_variable(
            name='_biases', initializer=tf.zeros(shape=[n_output_channels]))
        print(biases)
        conv = tf.nn.conv2d(input=input_tensor,
                            filter=weights,
                            strides=strides,
                            padding=padding_mode)
        print(conv)
        conv = tf.nn.bias_add(conv, biases, name='net_pre-activation')
        print(conv)
        conv = tf.nn.relu(conv, name='activation')
        print(conv)

        return conv
