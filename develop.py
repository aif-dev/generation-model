from models.music_transfomer import MusicTransformer
from data.factory import get_dataset
import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = ''

dataset = 'pianoroll'
data_path = 'data/datasets/pianoroll'
rundir = 'data'
input_shape = (100, 10)

dataset = get_dataset(dataset, data_path, rundir, input_shape, 2)
training_sequence, validation_sequence = dataset.create()

sample = training_sequence.take(1)
x, y = next(iter(sample))
# print(x)
# print(y)

model = MusicTransformer(embedding_dim=88, max_seq=100, debug=False, vocab_size=88)
