import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 分類器を構築するヘルパー関数
def build_classifier(data, labels, n_classes=2):
    data_shape = data.get_shape().as_list()
    weights = tf.get_variable(name='weights',
                              shape=(data_shape[1], n_classes),
                              dtype=tf.float32)
    bias = tf.get_variable(name='bias',
                           initializer=tf.zeros(shape=n_classes))
    logits = tf.add(tf.matmul(data, weights),
                    bias,
                    name='logits')
    return logits, tf.nn.softmax(logits)

# ジェネレータを構築するヘルパー関数
def build_generator(data, n_hidden):
    data_shape = data.get_shape().as_list()
    w1 = tf.Variable(tf.random_normal(shape=(data_shape[1], n_hidden)),
                     name='w1')
    b1 = tf.Variable(tf.zeros(shape=n_hidden),
                     name='b1')
    hidden = tf.add(tf.matmul(data, w1), b1, name='hidden_pre-activation')
    hidden = tf.nn.relu(hidden, 'hidden_activation')
    w2 = tf.Variable(tf.random_normal(shape=(n_hidden, data_shape[1])),
                     name='w2')
    b2 = tf.Variable(tf.zeros(shape=data_shape[1]),
                     name='b2')
    output = tf.add(tf.matmul(hidden, w2), b2, name = 'output')
    return output, tf.nn.sigmoid(output)

# 計算グラフの構築
batch_size = 64
g = tf.Graph()

with g.as_default():
    tf_X = tf.placeholder(shape=(batch_size, 100),
                          dtype=tf.float32,
                          name='tf_X')

    # ジェネレータを構築
    with tf.variable_scope('generator'):
        gen_out1 = build_generator(data=tf_X,
                                   n_hidden=50)

    # 分類器を構築
    with tf.variable_scope('classifier') as scope:
        # 元のデータに対する分類器
        cls_out1 = build_classifier(data=tf_X,
                                    labels=tf.ones(shape=batch_size))
        # 生成されたデータに対して分類器を再利用
        scope.reuse_variables()
        cls_out2 = build_classifier(data=gen_out1[1],
                                    labels=tf.zeros(shape=batch_size))
        init_op = tf.global_variables_initializer()
