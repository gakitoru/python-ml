import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
from setup_ch16_2 import *

class CharRNN(object):
    def __init__(self, num_classes, batch_size=64, num_steps=100,
                 lstm_size=128, num_layers=1, learning_rate=0.001,
                 keep_prob=0.5, grad_clip=5, sampling=False):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.grad_clip = grad_clip

        self.g = tf.Graph()
        with self.g.as_default():
            tf.set_random_seed(123)

            self.build(sampling=sampling)
            self.saver = tf.train.Saver()
            self.init_op = tf.global_variables_initializer()

    def build(self, sampling):
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size = self.batch_size
            num_steps = self.num_steps

        tf_x = tf.placeholder(tf.int32,
                              shape = [batch_size, num_steps],
                              name = 'tf_x')
        tf_y = tf.placeholder(tf.int32,
                              shape = [batch_size, num_steps],
                              name = 'tf_y')

        tf_keepprob = tf.placeholder(tf.float32,
                                     name='tf_keepprob')

        # one-hotエンコーディングを適用
        x_onehot = tf.one_hot(tf_x, depth=self.num_classes)
        y_onehot = tf.one_hot(tf_y, depth=self.num_classes)

        # 多層RNNのセルを構築
        cells = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size),
                output_keep_prob=tf_keepprob) for _ in range(self.num_layers)])

        # 初期状態を定義
        self.initial_state = cells.zero_state(batch_size, tf.float32)

        # RNNで各シーケンスステップを実行
        lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cells, x_onehot, initial_state=self.initial_state)

        print(' << lstm_outputs >> ', lstm_outputs)

        # 2次元テンソルに変形
        seq_output_reshaped = tf.reshape(lstm_outputs,
                                         shape=[-1, self.lstm_size],
                                         name='seq_output_reshaped')

        #　総入力を取得
        logits = tf.layers.dense(inputs=seq_output_reshaped,
                                 units=self.num_classes,
                                 activation=None,
                                 name='logits')

        # 次の文字バッチの確率　
        proba = tf.nn.softmax(logits, name='probabilities')

        # コスト関数を定義
        y_reshaped = tf.reshape(y_onehot,
                                shape=[-1, self.num_classes],
                                name='y_reshaped')

        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                    labels=y_reshaped),
            name='cost')

        # 勾配発散問題を回避するための勾配刈り込み
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          self.grad_clip)

        # オプティマイザを定義
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, tvars),
                                             name='train_op')


    def train(self, train_x, train_y, num_epochs, ckpt_dir='./model/'):
        # チェックポイントディレクトリがまだ存在しない場合は作成
        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)

        with tf.Session(graph=self.g) as sess:
            sess.run(self.init_op)

            n_batches = int(train_x.shape[1]/self.num_steps)
            iterations = n_batches * num_epochs
            for epoch in range(num_epochs):

                # ネットワークをトレーニング
                new_state = sess.run(self.initial_state)
                loss = 0

                # ミニバッチジェネレータ
                bgen = create_batch_generator(train_x, train_y, self.num_steps)
                for b, (batch_x, batch_y) in enumerate(bgen, 1):
                    iteration = epoch*n_batches + b

                    feed = {'tf_x:0': batch_x, 'tf_y:0': batch_y,
                            'tf_keepprob:0': self.keep_prob,
                            self.initial_state : new_state}

                    batch_cost, _, new_state = sess.run(
                        ['cost:0', 'train_op', self.final_state],
                        feed_dict=feed)

                    if iteration % 10 == 0:
                        print('Epoch %d/%d Iteration %d | Training loss: %.4f' % (epoch + 1, num_epochs, iteration, batch_cost))

                self.saver.save(sess,
                                os.path.join(ckpt_dir, 'language_modeling.ckpt'))

    def sample(self, output_length, ckpt_dir, starter_seq="The "):
        observed_seq = [ch for ch in starter_seq]
        with tf.Session(graph=self.g) as sess:
            self.saver.restore(sess,
                               tf.train.latest_checkpoint(ckpt_dir))

            # 1: starter_seqを使ってモデルを実行
            new_state = sess.run(self.initial_state)
            for ch in starter_seq:
                x = np.zeros((1, 1))
                x[0, 0] = char2int[ch]
                feed = {'tf_x:0' : x ,'tf_keepprob:0': 1.0,
                        self.initial_state:new_state}
                proba, new_state = sess.run(
                    ['probabilities:0', self.final_state],
                    feed_dict=feed)
            ch_id = get_top_char(proba, len(chars))
            observed_seq.append(int2char[ch_id])

            # 2: 更新されたobserved_seqを使ってモデルを実行
            for i in range(output_length):
                x[0, 0] = ch_id
                feed = {'tf_x:0':x, 'tf_keepprob:0': 1.0,
                        self.initial_state: new_state}

                proba, new_state = sess.run(
                    ['probabilities:0', self.final_state],
                    feed_dict=feed)

                ch_id = get_top_char(proba, len(chars))
                observed_seq.append(int2char[ch_id])

        return ''.join(observed_seq)

def get_top_char(probas, char_size, top_n=5):
    p = np.squeeze(probas)
    p[np.argsort(p)[:-top_n]] = 0.0
    p = p/np.sum(p)
    ch_id = np.random.choice(char_size, 1, p=p)[0]
    return ch_id
        
          
        
