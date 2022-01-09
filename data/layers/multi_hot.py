from typing import List
import tensorflow as tf


class MultiHot(tf.keras.layers.Layer):
    def __init__(self, notes_no: int = 88, sequence_length: int = 10):
        super(MultiHot, self).__init__()
        self._notes_no = notes_no
        self._sequence_length = sequence_length

    def call(self, input: List[List]):
        multihot_list = []

        for elem in input:
            if len(elem) > 0:
                multihot_tensor = tf.scatter_nd(indices=tf.transpose([elem]),
                                                updates=tf.tile([1.0], [len(elem)]),
                                                shape=[88])
                multihot_list.append(multihot_tensor)
            else:
                multihot_list.append(tf.zeros(88))

        return tf.convert_to_tensor(multihot_list)
