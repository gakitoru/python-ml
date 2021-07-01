from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
import tensorflow as tf

def build_rnn_model(embedding_dim, vocab_size, recurrent_type='SimpleRNN', n_recurrent_units=64,
                    n_recurrent_layers=1, bidirectional=True):
    tf.random.set_seed(1)
    ## モデルを構築
    model = tf.keras.Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                        name='embed-layer'))

    for i in range(n_recurrent_layers):
        return_sequences = (i < n_recurrent_layers-1)

        if recurrent_type == 'SimpleRNN':
            recurrent_layer = SimpleRNN(units=n_recurrent_units,
                                        return_sequences=return_sequences,
                                        name='simprnn-layer-{}'.format(i))
        elif recurrent_type == 'LSTM':
            recurrent_layer = LSTM(units=n_recurrent_units,
                                   return_sequences=return_sequences,
                                   name='lstm-layer-{}'.format(i))
        elif recurrent_type == 'GRU':
            recurrent_layer = GRU(units=n_recurrent_units,
                                  return_sequences=return_sequences,
                                  name='gru-layer-{}'.format(i))
        if bidirectional:
            recurrent_layer = Bidirectional(recurrent_layer,
                                            name='bidir-' + recurrent_layer.name)
        model.add(recurrent_layer)

    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    return model
              
