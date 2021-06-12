import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
from conv_layer import *
from fc_layer import *
from load_mnist import *

def build_cnn(learning_rate):
    # Xとyのプレースホルダを作成
    tf_x = tf.placeholder(tf.float32, shape=[None, 784], name='tf_x')
    tf_y = tf.placeholder(tf.int32, shape=[None], name='tf_y')

    # xを4次元テンソルに変換
    #  [バッチサイズ、幅、高さ、１]
    tf_x_image = tf.reshape(tf_x, shape=[-1, 28, 28, 1], name='tf_x_reshaped')
    # one-hotエンコーディング
    tf_y_onehot = tf.one_hot(indices=tf_y, depth=10, dtype=tf.float32,
                             name='tf_y_onehot')
    # 第1層: 畳み込み層1
    print('\nBuilding 1st layer: ')
    h1 = conv_layer(tf_x_image, name='conv_1',
                    kernel_size=(5, 5),
                    padding_mode='VALID',
                    n_output_channels=32)
    h1_pool = tf.nn.max_pool(h1,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME')

    # 第2層: 畳み込み層 2
    print('\nBuilding 2nd layer: ')
    h2 = conv_layer(h1_pool, name='conv_2',
                    kernel_size=(5, 5),
                    padding_mode='VALID',
                    n_output_channels=64)

    # 最大値プーリング
    h2_pool = tf.nn.max_pool(h2,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME')

    # 第3層:　全結合層
    print('\nBuilding 3rd layer:')
    h3 = fc_layer(h2_pool,
                  name='fc_3',
                  n_output_units=1024,
                  activation_fn=tf.nn.relu)

    # ドロップアウト
    keep_prob = tf.placeholder(tf.float32, name='fc_keep_prob')
    h3_drop = tf.nn.dropout(h3, keep_prob=keep_prob, name='dropout_layer')

    # 第4層: 全結合層2(線形活性化)
    print('\nBuilding 4th layer:')
    h4 = fc_layer(h3_drop,
                  name='fc_4',
                  n_output_units=10,
                  activation_fn=None)

    # 予測
    predictions = {
        'probabilities' : tf.nn.softmax(h4, name='probabilities'),
        'labels' : tf.cast(tf.argmax(h4, axis=1), tf.int32, name='labels')
    }

    # Tensorboadで計算グラフを可視化

    # 損失関数と最適化
    cross_entropy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=h4, labels=tf_y_onehot),
        name='cross_entropy_loss')

    # オプティマイザ
    optimizer = tf.train.AdamOptimizer(learning_rate)
    optimizer = optimizer.minimize(cross_entropy_loss, name='train_op')

    # 予測正解率を特定　
    correct_predictions = tf.equal(predictions['labels'], tf_y,
                                   name='correct_preds')
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32),
                              name='accuracy')

def save(saver, sess, epoch, path='./model/'):
    if not os.path.isdir(path):
        os.makedirs(path)

    print('Saving model in %s'  % path)
    saver.save(sess, os.path.join(path, 'cnn-model.ckpt'), global_step=epoch)

def load(saver, sess, path, epoch):
    print('Loading model from %s' % path)
    saver.restore(sess, os.path.join(path, 'cnn-model.ckpt-%d' % epoch))

def train(sess, training_set, validation_set=None, initialize=True,
          epochs=20, shuffle=True, dropout=0.5, random_seed=None):
    X_data = np.array(training_set[0])
    y_data = np.array(training_set[1])
    training_loss =[]

    # 変数を初期化
    if initialize:
        sess.run(tf.global_variables_initializer())

    # batch_generatorでシャッフルするため
    np.random.seed(random_seed)

    for epoch in range(1, epochs+1):
        batch_gen = batch_generator(X_data, y_data, shuffle=shuffle)
        avg_loss = 0.0
        for i, (batch_x, batch_y) in enumerate(batch_gen):
            feed = {'tf_x:0' : batch_x,
                    'tf_y:0' : batch_y,
                    'fc_keep_prob:0': dropout}

            loss, _ = sess.run(['cross_entropy_loss:0', 'train_op'],
                               feed_dict=feed)
            avg_loss += loss

        training_loss.append(avg_loss / (i+1))
        print('Epoch %02d Training Avg. Loss: %7.3f' %
              (epoch, avg_loss), end=' ')
        if validation_set is not None:
            feed = {'tf_x:0' : validation_set[0],
                    'tf_y:0' : validation_set[1],
                    'fc_keep_prob:0' : 1.0}
            valid_acc = sess.run('accuracy:0', feed_dict=feed)
            print(' Validation Acc: %7.3f' % valid_acc)
        else:
            print()


def predict(sess, X_test, return_proba=False):
    feed = {'tf_x:0' : X_test, 'fc_keep_prob:0' : 1.0}
    if return_proba:
        return sess.run('probabilities:0', feed_dict=feed)
    else:
        return sess.run('labels:0', feed_dict=feed)
