import numpy as np

with open('pg2265.txt', 'r', encoding='utf-8') as f:
    text = f.read()

text = text[15858:]
chars = set(text)
char2int = {ch:i for i, ch in enumerate(chars)}
int2char = dict(enumerate(chars))
text_ints = np.array([char2int[ch] for ch in text], dtype=np.int32)

def reshape_data(sequence, batch_size, num_steps):
    mini_batch_length = batch_size * num_steps
    num_batches = int(len(sequence) / mini_batch_length)

    if num_batches*mini_batch_length + 1 > len(sequence):
        num_bathces = num_bathces - 1

    x = sequence[0: num_batches*mini_batch_length]
    y = sequence[1: num_batches*mini_batch_length + 1]

    x_batch_splits = np.split(x, batch_size)
    y_batch_splits = np.split(y, batch_size)

    x = np.stack(x_batch_splits)
    y = np.stack(y_batch_splits)

    return x, y

def create_batch_generator(data_x, data_y, num_steps):
    batch_size, tot_batch_length = data_x.shape
    num_batches = int(tot_batch_length/num_steps)
    for b in range(num_batches):
        yield(data_x[:, b*num_steps: (b+1)*num_steps],
              data_y[:, b*num_steps: (b+1)*num_steps])

