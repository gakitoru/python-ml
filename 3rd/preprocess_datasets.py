from collections import Counter
import tensorflow as tf
import tensorflow_datasets as tfds


def preprocess_datasets(ds_raw_train, ds_raw_valid, ds_raw_test,
                        max_seq_length=None, batch_size=32):
    ## 手順2 : 一意なトークンを特定(手順1はすでに実行されている)
    tokenizer = tfds.deprecated.text.Tokenizer()
    token_counts = Counter()

    for example in ds_raw_train:
        tokens = tokenizer.tokenize(example[0].numpy()[0])
        if max_seq_length is not None:
            tokens = tokens[-max_seq_length:]
        token_counts.update(tokens)

    print('Vocab-size:', len(token_counts))

    ## 手順3 : テキストをエンコード
    encoder = tfds.deprecated.text.TokenTextEncoder(token_counts)
    def encode(text_tensor, label):
        text = text_tensor.numpy()[0]
        encoded_text = encoder.encode(text)
        if max_seq_length is not None:
            encoded_text = encoded_text[-max_seq_length:]
        return encoded_text, label

    def encode_map_fn(text, label):
        return tf.py_function(encode, inp=[text, label],
                              Tout=(tf.int64, tf.int64))

    ds_train = ds_raw_train.map(encode_map_fn)
    ds_valid = ds_raw_valid.map(encode_map_fn)
    ds_test = ds_raw_test.map(encode_map_fn)

    ## 手順4 : データセットをバッチ分割
    train_data = ds_train.padded_batch(batch_size, padded_shapes=([-1], []))
    valid_data = ds_valid.padded_batch(batch_size, padded_shapes=([-1], []))
    test_data = ds_test.padded_batch(batch_size, padded_shapes=([-1], []))

    return (train_data, valid_data, test_data, len(token_counts))
