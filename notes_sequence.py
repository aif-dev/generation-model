import math
from keras.utils import Sequence, np_utils
import numpy as np


class NotesSequence(Sequence):
    def __init__(self, notes, batch_size, sequence_length, prediction_size):
        self.notes = notes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.prediction_size = prediction_size

    def __len__(self):
        return math.floor(len(self.notes[: -self.sequence_length]) / self.batch_size)

    def __getitem__(self, idx):
        network_input = []
        network_output = []
        for i in range(self.batch_size):
            sequence_in = self.notes[
                idx * self.batch_size
                + i : idx * self.batch_size
                + i
                + self.sequence_length
            ]
            note_out = self.notes[idx * self.batch_size + i + self.sequence_length]

            network_input.append([note for note in sequence_in])
            network_output.append(note_out)

        n_patterns = len(network_input)

        network_input = np.reshape(
            network_input, (n_patterns, self.sequence_length, self.prediction_size)
        )

        return np.array(network_input), np.array(network_output)
