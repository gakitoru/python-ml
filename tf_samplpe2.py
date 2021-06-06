import tensorflow as tf
import numpy as np

g = tf.Graph()

with g.as_default():
    x = tf.placeholder(dtype=tf.float32, shape=(None, 2, 3), name='input_x')
    x2 = tf.reshape(x, shape=(-1, 6), name='x2')

    ## 各列の合計を求める
    xsum = tf.reduce_sum(x2, axis=0, name='col_sum')

    ## 各列の平均を求める
    xmean = tf.reduce_mean(x2, axis=0, name='col_mean')

with tf.Session(graph=g) as sess:
    x_array = np.arange(18).reshape(3, 2, 3)
    print('input shape: ', x_array.shape)
    print('Reshaped:\n', sess.run(x2, feed_dict={x:x_array}))
    print('Column Sums:\n', sess.run(xsum, feed_dict={x:x_array}))
    print('Column Mean:\n', sess.run(xmean, feed_dict={x:x_array}))
