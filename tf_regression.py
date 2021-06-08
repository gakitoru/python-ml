import tensorflow as tf
import numpy as np

g = tf.Graph()

with g.as_default():
    tf.set_random_seed(123)

    # プレースホルダを定義
    tf_x = tf.placeholder(shape=(None), dtype=tf.float32, name='tf_x')
    tf_y = tf.placeholder(shape=(None), dtype=tf.float32, name='tf_y')

    # 変数(モデルのパラメータ) を定義
    weight = tf.Variable(tf.random_normal(shape=(1, 1), stddev=0.25),
                         name='weight')
    bias = tf.Variable(0.0, name='bias')

    # モデルを構築
    y_hat = tf.add(weight * tf_x, bias, name='y_hat')

    # コストを計算
    cost = tf.reduce_mean(tf.square(tf_y - y_hat), name='cost')

    # モデルをトレーニング
    optim = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optim.minimize(cost, name='train_op')
