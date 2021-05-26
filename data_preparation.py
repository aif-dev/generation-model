import glob
import os
import sys
import datetime
import pickle
from pprint import pprint
from multiprocessing import Pool, cpu_count
import checksumdir
import numpy as np
from music21 import converter, instrument, stream, note, chord
from keras.utils import np_utils
from random_word import RandomWords


MIDI_SONGS_DIR = "midi_songs"
DATA_DIR = "data"
NOTES_FILENAME = "notes"
VOCABULARY_FILENAME = "vocabulary"
HASH_FILENAME = "dataset_hash"
RESULTS_DIR = "results"
SEQUENCE_LENGTH = 100

"""
changing this value requires refactoring

predict.py -> loop inside generate_notes() [getting prediction]
data_preparation.py -> loop inside prepare_sequences_for_training() [out sequences]
"""
NUM_NOTES_TO_PREDICT = 1


def save_data_hash(hash_value):
    if not os.path.isdir(DATA_DIR):
        os.mkdir(DATA_DIR)

    hash_file_path = os.path.join(DATA_DIR, HASH_FILENAME)
    with open(hash_file_path, "wb") as hash_file:
        pickle.dump(hash_value, hash_file)


def is_data_changed():
    current_hash = checksumdir.dirhash(MIDI_SONGS_DIR)

    hash_file_path = os.path.join(DATA_DIR, HASH_FILENAME)
    if not os.path.exists(hash_file_path):
        save_data_hash(current_hash)
        return True

    with open(hash_file_path, "rb") as hash_file:
        previous_hash = pickle.load(hash_file)

    if previous_hash != current_hash:
        save_data_hash(current_hash)
        return True

    return False


def get_notes_from_file(file):
    notes = []
    midi = converter.parse(file)

    print(f"Parsing {file}")

    try:
        # file has instrument parts
        instrument_stream = instrument.partitionByInstrument(midi)
        notes_to_parse = instrument_stream.parts[0].recurse()
    except:
        # file has notes in a flat structure
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append(".".join(str(n) for n in element.normalOrder))

    return notes


def get_notes_from_dataset():
    """Get all the notes and chords from the midi files in the ./midi_songs directory"""

    notes_path = os.path.join(DATA_DIR, NOTES_FILENAME)
    notes = []
    if is_data_changed():
        try:
            with Pool(cpu_count() - 1) as pool:
                notes_from_files = pool.map(
                    get_notes_from_file, glob.glob(f"{MIDI_SONGS_DIR}/*.mid")
                )

                for notes_from_file in notes_from_files:
                    for note in notes_from_file:
                        notes.append(note)

            with open(notes_path, "wb") as notes_data_file:
                pickle.dump(notes, notes_data_file)

        except:
            hash_file_path = os.path.join(DATA_DIR, HASH_FILENAME)
            os.remove(hash_file_path)
            print("Removed the hash file")
            sys.exit(1)

    else:
        with open(notes_path, "rb") as notes_data_file:
            notes = pickle.load(notes_data_file)

    return notes


def create_vocabulary_for_training(notes):
    print("*** Creating new vocabulary ***")

    # these are either notes or chords
    sound_names = sorted(set(item for item in notes))
    vocab = dict((note, number) for number, note in enumerate(sound_names))

    vocab_path = os.path.join(DATA_DIR, VOCABULARY_FILENAME)
    with open(vocab_path, "wb") as vocab_data_file:
        pickle.dump(vocab, vocab_data_file)

    return vocab


def load_vocabulary_from_training():
    print("*** Restoring vocabulary used for training ***")

    vocab_path = os.path.join(DATA_DIR, VOCABULARY_FILENAME)
    with open(vocab_path, "rb") as vocab_data_file:
        return pickle.load(vocab_data_file)


def prepare_sequences_for_training(notes, vocab, vocab_size):
    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(len(notes) - SEQUENCE_LENGTH):
        sequence_in = notes[i : i + SEQUENCE_LENGTH]
        note_out = notes[i + SEQUENCE_LENGTH]
        network_input.append([vocab[note] for note in sequence_in])
        network_output.append(vocab[note_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    unnormalized_network_input = np.reshape(
        network_input, (n_patterns, SEQUENCE_LENGTH, NUM_NOTES_TO_PREDICT)
    )
    normalized_network_input = unnormalized_network_input / float(vocab_size)

    network_output = np_utils.to_categorical(network_output)

    return (normalized_network_input, network_output)


def prepare_sequences_for_prediction(notes, vocab):
    network_input = []
    for i in range(len(notes) - SEQUENCE_LENGTH):
        sequence_in = notes[i : i + SEQUENCE_LENGTH]
        network_input.append([vocab[note] for note in sequence_in])

    return network_input


def save_midi_file(prediction_output):
    """convert the output from the prediction to notes and create a midi file
    from the notes"""
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
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
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    random_words = RandomWords().get_random_words()
    output_name = ""
    for i in range(2):
        output_name += random_words[i] + "_"
    output_name = output_name.rstrip("_").lower()

    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp=f"{RESULTS_DIR}/{output_name}.mid")
