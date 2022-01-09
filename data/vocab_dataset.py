import glob
import os
import sys
import pickle
import math
import datetime
from typing import List, Tuple, Dict

import tensorflow as tf
import numpy as np
from multiprocessing import Pool, cpu_count
from music21 import converter, instrument, stream, note, chord
from random_word import RandomWords
from collections import Counter
from .base_dataset import BaseDataset, BUFFER_SIZE

"""
changing this value requires refactoring

predict.py -> loop inside generate_notes() [getting prediction]
data_preparation.py -> loop inside prepare_sequences_for_training() [out sequences]
"""
NUM_NOTES_TO_PREDICT = 1


class VocabDataset(BaseDataset):
    """Dataset representing data with tokens"""

    def __init__(self, **kwargs):
        super().__init__(notes_name="vds_notes", vocab_name="vds_vocabulary", **kwargs)
        self.vocab = None
        self.notes = None
        self.vocab_size = 0

    def parse_file(self, file: str) -> List[str]:
        print(f"Parsing {file}")

        notes = []

        # parsing a midi file
        midi = converter.parse(file)

        # grouping based on different instruments
        s2 = instrument.partitionByInstrument(midi)

        # Looping over all the instruments
        for part in s2.parts:

            # select elements of only piano
            if "Piano" in str(part):

                notes_to_parse = part.recurse()

                # finding whether a particular element is note or a chord
                for element in notes_to_parse:

                    # note
                    if isinstance(element, note.Note):
                        notes.append(str(element.name))

                    # chord
                    elif isinstance(element, chord.Chord):
                        notes.append(".".join(str(n) for n in element.normalOrder))
        return notes

    def get_notes_from_dataset(self) -> None:
        notes_path = os.path.join(self.rundir, self.notes_name)
        notes = []

        if self.is_data_changed(self.data_path):
            try:
                with Pool(cpu_count() - 1) as pool:
                    notes_from_files = pool.map(self.parse_file, glob.glob(f"{self.data_path}/*.mid"))

                notes_ = []

                for notes_from_file in notes_from_files:
                    for note in notes_from_file:
                        notes_.append(note)

                freq = dict(Counter(notes_))
                frequent_notes = [note_ for note_, count in freq.items() if count >= 64 or len(note_) == 1]

                for note in notes_:
                    if note in frequent_notes:
                        notes.append(note)

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

    def create_vocab(self) -> None:
        print("*** Creating new vocabulary ***")

        # these are either notes or chords
        sound_names = sorted(set(item for item in self.notes))
        vocab = dict((note, number) for number, note in enumerate(sound_names))

        vocab_path = os.path.join(self.rundir, self.vocab_name)
        with open(vocab_path, "wb") as vocab_data_file:
            pickle.dump(vocab, vocab_data_file)

        print(f"*** Vocabulary size {len(vocab)}")

        self.vocab = vocab
        self.vocab_size = len(vocab)

    def create_sequences(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Creates windows and corresponding labels from input sequence"""
        seq_len = sum(self.input_shape)

        windows = dataset.window(seq_len, shift=1, stride=1, drop_remainder=True)

        flatten = lambda x: x.batch(seq_len, drop_remainder=True)
        sequences = windows.flat_map(flatten)

        def scale_notes(x):
            x = x / self.vocab_size
            return x

        def split_labels(sequences):
            inputs = sequences[: self.input_shape[0]]
            label = sequences[-1]

            return scale_notes(inputs), tf.one_hot(label, self.vocab_size)

        return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

    def create_training(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Creates data loaders for training and validation sets"""
        self.get_notes_from_dataset()
        self.create_vocab()

        training_split = 0.8
        dataset_split = math.ceil(training_split * len(self.notes))

        self.notes = [self.vocab[note] for note in self.notes]

        train_ds = tf.data.Dataset.from_tensor_slices(self.notes[:dataset_split])
        val_ds = tf.data.Dataset.from_tensor_slices(self.notes[dataset_split:])

        train_sequence = self.create_sequences(train_ds)
        val_sequence = self.create_sequences(val_ds)

        return train_sequence.shuffle(BUFFER_SIZE).batch(self.batch_size, drop_remainder=True).cache().prefetch(
            tf.data.experimental.AUTOTUNE
        ), val_sequence.shuffle(BUFFER_SIZE).batch(self.batch_size, drop_remainder=True).cache().prefetch(
            tf.data.experimental.AUTOTUNE
        )

    def create_test(self):
        #TO-DO: change to support more than one file in dierectory
        file = glob.glob(f"{self.data_path}/*.mid")[0]
        notes = self.parse_file(file)
        self.load_vocabulary_from_training()

        if len(notes) < self.input_shape[0]:
            print(f"File is to short. Min length: {self.input_shape[0]} sounds, provided: {len(notes)}.")
            sys.exit(1)

        sequence_in = notes[: self.input_shape[0]]

        x = np.array([self.get_best_representation(self.vocab, sound) for sound in sequence_in]).reshape(
            self.input_shape[0], 1
        )
        x = x / float(self.vocab_size)

        return x

    def create(self, mode="train"):
        if mode == "train":
            return self.create_training()
        elif mode == "test":
            return self.create_test()
        else:
            raise ValueError("Not a valid mode")

    def get_best_representation(self, vocab: Dict[str, int], pattern: str) -> int:
        """assumption: all 12 single notes are present in vocabulary"""
        if pattern in vocab.keys():
            return vocab[pattern]

        chord_sounds = [int(sound) for sound in pattern.split(".")]
        unknown_chord = chord.Chord(chord_sounds)
        root_note = unknown_chord.root()
        print(f"*** Mapping {unknown_chord} to {root_note} ***")
        return vocab[root_note.name]

    def generate_midi(self, prediction_output, save_path):
        offset = 0
        output_notes = []
        inverted_vocab = {a: b for b, a in self.vocab.items()}
        mapped_output = [inverted_vocab[item] for item in prediction_output]

        # create note and chord objects based on the values generated by the model
        for pattern in mapped_output:
            # pattern is a chord
            if ("." in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split(".")
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            # pattern is a note
            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)

            # increase offset each iteration so that notes do not stack
            offset += 0.5

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

    def load_vocabulary_from_training(self):
        print("*** Restoring vocabulary used for training ***")

        vocab_path = os.path.join(self.rundir, self.vocab_name)
        with open(vocab_path, "rb") as vocab_data_file:
            self.vocab = pickle.load(vocab_data_file)
            self.vocab_size = len(self.vocab)
