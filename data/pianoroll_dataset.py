import glob
import os
import sys
import pickle
import math
import datetime
import tensorflow as tf
import numpy as np
from multiprocessing import Pool, cpu_count
from music21 import converter, instrument, stream, note, chord
from random_word import RandomWords
from collections import Counter
from base_dataset import BaseDataset, BUFFER_SIZE

"""
changing this value requires refactoring

predict.py -> loop inside generate_notes() [getting prediction]
data_preparation.py -> loop inside prepare_sequences_for_training() [out sequences]
"""
NUM_NOTES_TO_PREDICT = 1


class PianoRollDataset(BaseDataset):
    """Dataset representing data with tokens"""

    def __init__(self, **kwargs):
        super().__init__(notes_name="Piano", vocab_name="PianoRoll", **kwargs)
        self.pianoroll_vocab = None
        self.pianorolls = None
        self.pianoroll_size = 0

    # Function to parse a midi file
    def parse_file(self, file: str) -> list[str]:
        print(f"Parsing {file}")
        pianoroll = []
        # parsing a midi file
        midi = converter.parse(file)
        for part in midi.parts:
            instrument = part.getInstrument().instrumentName
            if instrument == "Piano":
                for note in part.flat.notes:
                    if note.isChord:
                        start = note.offset
                        duration = note.quarterLength
                        for chord_note in note.pitches:
                            pitch = chord_note.ps
                            volume = note.volume.realized
                            name = chord_note.nameWithOctave
                            pianoroll.append([start, duration, pitch, volume, name])
                else:
                    start = note.offset
                    duration = note.quarterLength
                    pitch = note.pitch.ps
                    volume = note.volume.realized
                    name = note.nameWithOctave
                pianoroll.append([start, duration, pitch, volume, name])
            pianoroll = sorted(pianoroll, key=lambda x: (x[0], x[2]))
            return pianoroll

    # Get Piano Roll from the dataset
    def get_pianoroll_from_dataset(self) -> None:
        pianorolls_path = os.path.join(self.rundir, self.notes_name)
        pianorolls = []

        if self.is_data_changed(self.data_path):
            try:
                with Pool(cpu_count() - 1) as pool:
                    pianorolls_from_files = pool.map(self.parse_file, glob.glob(f"{self.data_path}/*.mid"))

                pianorolls_ = []

                for pianorolls_from_file in pianorolls_from_files:
                    for pianoroll in pianorolls_from_file:
                        pianorolls_.append(pianoroll)

                freq = dict(Counter(pianorolls_))
                frequent_pianorolls = [pianorolls_ for pianorolls_, count in freq.items() if count >= 64 or len(pianorolls_) == 1]

                for pianorolls in pianorolls_:
                    if pianoroll in frequent_pianorolls:
                        pianorolls.append(pianoroll)

                with open(pianorolls_path, "wb") as pianorolls_data_file:
                    pickle.dump(pianorolls, pianorolls_data_file)

            except:
                hash_file_path = os.path.join(self.rundir, self.hash_filename)
                os.remove(hash_file_path)
                print("Removed the hash file")
                sys.exit(1)

        else:
            with open(pianorolls_path, "rb") as pianorolls_data_file:
                pianorolls = pickle.load(pianorolls_data_file)

        self.pianorolls = pianorolls

    def create_PianoRoll_Vocab(self) -> None:
        print("*** Creating new PianoRoll Vocab ***")

       #Pianorolls vocab
        sound_names = sorted(set(item for item in self.pianorolls))
        proll = dict((pianoroll, number) for number, pianoroll in enumerate(sound_names))

        pianoroll_path = os.path.join(self.rundir, self.vocab_name)
        with open(pianoroll_path, "wb") as pianoroll_data_file:
            pickle.dump(proll, pianoroll_data_file)

        print(f"*** PianoRoll size {len(proll)}")

        self.pianoroll_vocab = proll
        self.pianoroll_size = len(proll)

    # Create Sequence of Piano Roll#
    def create_sequences(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Creates windows and corresponding labels from input sequence"""
        seq_len = sum(self.input_shape)

        windows = dataset.window(seq_len, shift=1, stride=1, drop_remainder=True)

        flatten = lambda x: x.batch(seq_len, drop_remainder=True)
        sequences = windows.flat_map(flatten)

        def scale_notes(x):
            x = x / self.pianoroll_size
            return x

        def split_labels(sequences):
            inputs = sequences[: self.input_shape[0]]
            label = sequences[-1]

            return scale_notes(inputs), tf.one_hot(label, self.pianoroll_size)

        return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

    def create_training(self) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """Creates data loaders for training and validation sets"""
        self.get_pianoroll_from_dataset()
        self.create_PianoRoll_Vocab()

        training_split = 0.8
        dataset_split = math.ceil(training_split * len(self.pianorolls))

        self.pianorolls = [self.pianoroll_vocab[pianoroll] for pianoroll in self.pianorolls]

        train_ds = tf.data.Dataset.from_tensor_slices(self.pianorolls[:dataset_split])
        val_ds = tf.data.Dataset.from_tensor_slices(self.pianorolls[dataset_split:])

        train_sequence = self.create_sequences(train_ds)
        val_sequence = self.create_sequences(val_ds)

        return train_sequence.shuffle(BUFFER_SIZE).batch(self.batch_size, drop_remainder=True).cache().prefetch(
            tf.data.experimental.AUTOTUNE
        ), val_sequence.shuffle(BUFFER_SIZE).batch(self.batch_size, drop_remainder=True).cache().prefetch(
            tf.data.experimental.AUTOTUNE
        )

    def create_test(self):
        # TO-DO: change to support more than one file in dierectory
        file = glob.glob(f"{self.data_path}/*.mid")[0]
        pianorolls = self.parse_file(file)
        self.load_pianoroll_Vocab_from_training()

        if len(pianorolls) < self.input_shape[0]:
            print(f"File is to short. Min length: {self.input_shape[0]} sounds, provided: {len(pianorolls)}.")
            sys.exit(1)

        sequence_in = pianorolls[: self.input_shape[0]]

        x = np.array([self.get_best_representation(self.pianoroll_vocab, sound) for sound in sequence_in]).reshape(
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

    def get_best_representation(self, vocab: dict[str, int], pattern: str) -> int:
        """assumption: all 12 single notes are present in vocabulary"""
        if pattern in vocab.keys():
            return vocab[pattern]

        chord_sounds = [int(sound) for sound in pattern.split(".")]
        unknown_chord = chord.Chord(chord_sounds)
        root_note = unknown_chord.root()
        print(f"*** Mapping {unknown_chord} to {root_note} ***")
        return vocab[root_note.name]

    def generate_midi(self,prediction_output, save_path):
        offset = 0
        output_notes = []
        inverted_piano_vocab = {a: b for b, a in self.pianoroll_vocab.items()}
        mapped_output = [inverted_piano_vocab[item] for item in prediction_output]

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

    def load_pianoroll_Vocab_from_training(self):
        print("*** Restoring pianoroll_vocab used for training ***")

        pianoroll_vocab_path = os.path.join(self.rundir, self.vocab_name)
        with open(pianoroll_vocab_path, "rb") as pianoroll_vocab_data_file:
            self.pianoroll_vocab = pickle.load(pianoroll_vocab_data_file)
            self.vocab_size = len(self.Pianoroll_vocab)
