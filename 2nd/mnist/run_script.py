learning_rate=1e-4
random_seed=123
from build_cnn import *
from load_mnist import *
X_data, y_data = load_mnist('./', kind='train')
X_test, y_test = load_mnist('./', kind='t10k')
X_train, y_train = X_data[:50000, :], y_data[:50000]
X_valid, y_valid = X_data[50000:, :], y_data[50000:]
mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)
X_train_centered = (X_train - mean_vals) / std_val
X_valid_centered = X_valid - mean_vals
X_test_centered = (X_test - mean_vals) /std_val
g = tf.Graph()
with g.as_default():
    tf.set_random_seed(random_seed)
    build_cnn(learning_rate)
    saver = tf.train.Saver()
