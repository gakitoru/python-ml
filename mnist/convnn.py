import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from load_mnist import *

class ConvNN(object):
    def __init__(self, batchsize=64, epochs=20, learning_rate=1e-4,
                 dropout_rate=0.5, shuffle=True, random_seed=None):
        np.random.seed(random_seed)
        self.batchsize = batchsize
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.shuffle = shuffle

        g = tf.Graph()
        with g.as_default():
            # 乱数シードを設定
            tf.set_random_seed(random_seed)
            # モデルを構築
            self.build()
            # 変数を初期化
            self.init_op = tf.global_variables_initializer()
            # saver
            self.saver = tf.train.Saver()

        self.sess = tf.Session(graph=g)

    def build(self):
        # Xとyのプレースホルダを作成
        tf_x = tf.placeholder(tf.float32, shape=[None, 784], name='tf_x')
        tf_y = tf.placeholder(tf.int32, shape=[None], name='tf_y')
        is_train = tf.placeholder(tf.bool, shape=(), name='is_train')

        # xを4次元テンソルに変換 : [バッチサイズ, 幅, 高さ, 1]
        tf_x_image = tf.reshape(tf_x, shape=[-1, 28, 28, 1],
                                name='input_x_2dimages')

        tf_y_onehot = tf.one_hot(indices=tf_y, depth=10, dtype=tf.float32,
                                 name='input_y_onehot')
        
        # 第1層 : 畳み込み層1
        h1 = tf.layers.conv2d(tf_x_image,
                              kernel_size=(5, 5),
                              filters=32,
                              activation=tf.nn.relu)

        # 最大値プーリング
        h1_pool = tf.layers.max_pooling2d(h1,
                                          pool_size=(2, 2),
                                          strides=(2, 2))

        # 第2層 : 畳み込み層2
        h2 = tf.layers.conv2d(h1_pool,
                              kernel_size=(5, 5),
                              filters=64,
                              activation=tf.nn.relu)

        # 最大値プーリング
        h2_pool = tf.layers.max_pooling2d(h2,
                                          pool_size=(2, 2),
                                          strides=(2, 2))

        # 第3層 : 全結合層1
        input_shape = h2_pool.get_shape().as_list()
        n_input_units = np.prod(input_shape[1:])
        h2_pool_flat = tf.reshape(h2_pool, shape=[-1, n_input_units])
        h3 = tf.layers.dense(h2_pool_flat, 1024, activation=tf.nn.relu)

        # ドロップアウト
        h3_drop = tf.layers.dropout(h3,
                                    rate=self.dropout_rate,
                                    training=is_train)

        # 第4層 : 全結合層2 (線形活性化)
        h4 = tf.layers.dense(h3_drop,
                             10,
                             activation=None)

        # 予測
        predictions = {
            'probabilities' : tf.nn.softmax(h4, name='probabilities'),
            'labels' : tf.cast(tf.argmax(h4, axis=1), tf.int32, name='labels')
        }

        # 損失関数と最適化
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=h4, labels=tf_y_onehot),
            name='cross_entropy_loss')

        # オプティマイザ
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        optimizer = optimizer.minimize(cross_entropy_loss, name='train_op')

        # 予測正解率を特定
        correct_predictions = tf.equal(predictions['labels'], tf_y,
                                       name='correct_preds')

        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32),
                                  name='accuracy')

    def save(self, epoch, path='./tflayers-model/'):
        if not os.path.isdir(path):
            os.makedirs(path)

        print('Saving model in %s' % path)
        self.saver.save(self.sess,
                        os.path.join(path, 'model.ckpt'),
                        global_step=epoch)

    def load(self, epoch, path):
        print('Loading model from %s' % path)
        self.saver.restore(self.sess,
                           os.path.join(path, 'model.ckpt-%d' % epoch))

    def train(self, training_set, validation_set=None, initializer=True):
        # 変数を初期化
        if initializer:
            self.sess.run(self.init_op)

        self.train_cost_ = []
        X_data = np.array(training_set[0])
        y_data = np.array(training_set[1])

        for epoch in range(1, self.epochs + 1):
            batch_gen = batch_generator(X_data, y_data, shuffle=self.shuffle)
            avg_loss = 0.0
            for i, (batch_x, batch_y) in enumerate(batch_gen):
                feed = {'tf_x:0' : batch_x,
                        'tf_y:0' : batch_y,
                        'is_train:0' : True} # ドロップアウト
                loss, _ = self.sess.run(['cross_entropy_loss:0', 'train_op'],
                                        feed_dict=feed)
                avg_loss += loss

            print('Epoch %02d: Training Avg. Loss: %7.3f' %
                  (epoch, avg_loss), end=' ')
            if validation_set is not None:
                feed = {'tf_x:0' : batch_x,
                        'tf_y:0' : batch_y,
                        'is_train:0' : False} # ドロップアウト
                valid_acc = self.sess.run('accuracy:0', feed_dict=feed)
                print('Validation Acc: %7.3f' % valid_acc)
            else:
                print()

    def predict(self, X_test, return_proba=False):
        feed = {'tf_x:0' : X_test,
                'is_train:0' : False}

        if return_proba:
            return self.sess.run('probabilities:0', feed_dict=feed)
        else:
            return self.sess.run('labels:0', feed_dict=feed)
        
        

        
        
                                          
