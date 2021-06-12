import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

def fc_layer(input_tensor, name, n_output_units, activation_fn=None):
    with tf.variable_scope(name):
        input_shape = input_tensor.get_shape().as_list()[1:]
        n_input_units = np.prod(input_shape)
        if len(input_shape) > 1:
            input_tensor = tf.reshape(input_tensor, shape=(-1, n_input_units))

        weights_shape = [n_input_units, n_output_units]
        weights = tf.get_variable(name='_weights', shape=weights_shape)
        print(weights)
        biases = tf.get_variable(name='biases', initializer=tf.zeros(shape=[n_output_units]))
        print(biases)

        layer = tf.matmul(input_tensor, weights)
        print(layer)
        layer = tf.nn.bias_add(layer, biases, name='net_pre-activation')
        print(layer)
        if activation_fn is None:
            return layer
        layer = activation_fn(layer, name='activation')
        print(layer)
        return layer
