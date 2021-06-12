import tensorflow as tf
import numpy as np

class TfLinreg(object):

    def __init__(self, x_dim, learning_rate=0.01, random_seed=None):
        self.x_dim = x_dim
        self.learning_rate = learning_rate
        self.g = tf.Graph()

        ## モデルを構築
        with self.g.as_default():
            ## グラフレベルの乱数シードを設定
            tf.set_random_seed(random_seed)

            self.build()
            ## 変数のイニシャライザを作成
            self.init_op = tf.global_variables_initializer()

    def build(self):
        ## 入力用のプレースホルダを定義
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, self.x_dim), name='x_input')
        self.y = tf.placeholder(dtype=tf.float32, shape=(None), name='y_input')

        print(self.X)
        print(self.y)

        ## 重み行列とバイアスベクトルを定義
        w = tf.Variable(tf.zeros(shape=(1)), name='weight')
        b = tf.Variable(tf.zeros(shape=(1)), name='bias')
        print(w)
        print(b)

        self.z_net = tf.squeeze(w * self.X + b, name='z_net')
        print(self.z_net)

        sqr_errors = tf.square(self.y - self.z_net, name='sqr_errors')
        print(sqr_errors)
        self.mean_cost = tf.reduce_mean(sqr_errors, name='mean_cost')

        ## オプティマイザを作成　
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate, name='GradientDescent')
        self.optimizer = optimizer.minimize(self.mean_cost)
