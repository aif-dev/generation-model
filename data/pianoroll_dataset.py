import glob
import os
import sys
import pickle
import math
import datetime
from typing import Tuple

import tensorflow as tf
import numpy as np
from pretty_midi import PrettyMIDI
from multiprocessing import Pool, cpu_count
from music21 import instrument, stream, note, chord
from random_word import RandomWords
from .base_dataset import BaseDataset, BUFFER_SIZE

MAX_VOLUME = 128

class PianorollDataset(BaseDataset):
    """Dataset representing data with tokens"""

    def __init__(self, **kwargs):
        super().__init__(notes_name="pianoroll_notes", vocab_name="dummy", **kwargs)

    def _parse_file(self, file: str) -> np.ndarray:
        pm = PrettyMIDI(file)
        ins = pm.instruments[0]
        return ins.get_piano_roll(fs=2)[21:109, :].T

    def get_pianorolls(self) -> None:
        notes_path = os.path.join(self.rundir, self.notes_name)

        if self.is_data_changed(self.data_path):
            try:
                with Pool(cpu_count() - 1) as pool:
                    notes = pool.map(self.parse_file, glob.glob(f"{self.data_path}/*.mid"))

                with open(notes_path, "wb") as notes_data_file:
                    pickle.dump(notes, notes_data_file)

            except:
                hash_file_path = os.path.join(self.rundir, self.hash_filename)
                os.remove(hash_file_path)
                print("Removed the hash file")
                sys.exit(1)

        else:
            with open(notes_path, "rb") as notes_data_file:
                notes = pickle.load(notes_data_file)

        self.notes = notes

    def create_sequences(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Creates windows and corresponding labels from input sequence"""
        seq_len = sum(self.input_shape)

        datasets = []

        for elem in dataset:
            dataset = tf.data.Dataset.from_tensor_slices(elem)
            windows = dataset.window(seq_len, shift=1, stride=1, drop_remainder=True)
            flatten = lambda x: x.batch(seq_len, drop_remainder=True)
            datasets.append(windows.flat_map(flatten))

        # TO-DO: find more optimal way of concatenating multiple datasets
        sequences = datasets[0]
        for elem in datasets[1:]:
            sequences = sequences.concatenate(elem)

        def scale_notes(x):
            x = x / MAX_VOLUME
            return x

        def split_labels(sequences):
            inputs = sequences[: self.input_shape[0]]
            label = sequences[self.input_shape[0] :]

            return scale_notes(inputs), scale_notes(label)

        return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

    def create_training(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Creates data loaders for training and validation sets"""
        self.get_pianorolls()

        training_split = 0.8
        dataset_split = math.ceil(training_split * len(self.notes))

        train_rolls = self.notes[:dataset_split]
        val_rolls = self.notes[dataset_split:]

        train_sequence = self.create_sequences(train_rolls)
        val_sequence = self.create_sequences(val_rolls)

        return train_sequence.shuffle(BUFFER_SIZE).batch(self.batch_size, drop_remainder=True).cache().prefetch(
            tf.data.experimental.AUTOTUNE
        ), val_sequence.shuffle(BUFFER_SIZE).batch(self.batch_size, drop_remainder=True).cache().prefetch(
            tf.data.experimental.AUTOTUNE
        )

    def create_test(self):
        # TO-DO: change to support more than one file in dierectory
        file = glob.glob(f"{self.data_path}/*.mid")[0]
        notes = self.parse_file(file)

        if len(notes) < self.input_shape[0]:
            print(f"File is to short. Min length: {self.input_shape[0]} sounds, provided: {len(notes)}.")
            sys.exit(1)

        sequence_in = notes[: self.input_shape[0]]


        sequence_in = sequence_in / MAX_VOLUME

        return sequence_in

    def create(self, mode="train"):
        if mode == "train":
            return self.create_training()
        elif mode == "test":
            return self.create_test()
        else:
            raise ValueError("Not a valid mode")

    def generate_midi(self, prediction_output, save_path):
        offset = 0
        output_notes = []

        for col in prediction_output:
            notes = np.nonzero(col)[0]
            velocites = col[notes]

            if len(notes) == 1:
                new_note = note.Note(notes[0] + 21)
                new_note.volume = velocites[0] * MAX_VOLUME
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)

            elif len(notes) > 1:
                chord_notes = []
                for c_note, v in zip(notes, velocites):
                    new_note = note.Note(c_note + 21)
                    new_note.volume = v * MAX_VOLUME
                    new_note.storedInstrument = instrument.Piano()
                    chord_notes.append(new_note)
                new_chord = chord.Chord(chord_notes)
                new_chord.offset = offset
                output_notes.append(new_chord)

            offset += 1

        output_name = ""
        try:
            random_words = RandomWords().get_random_words()
            for i in range(2):
                output_name += random_words[i] + "_"
            output_name = output_name.rstrip("_").lower()

        except:
            output_name = f"output_{datetime.datetime.now()}"

        midi_stream = stream.Stream(output_notes)
        os.makedirs(save_path, exist_ok=True)
        fp = f"{save_path}/{output_name}.mid"
        print(f"saving {fp}")
        midi_stream.write("midi", fp=fp)
