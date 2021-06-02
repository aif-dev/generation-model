import glob
import os
import sys
import pickle
import math
import datetime
import shutil
from multiprocessing import Pool, cpu_count
import checksumdir
from music21 import converter, instrument, stream, note, chord
from random_word import RandomWords
from notes_sequence import NotesSequence


CHECKPOINTS_DIR = "checkpoints"
MIDI_SONGS_DIR = "midi_songs"
DATA_DIR = "data"
NOTES_FILENAME = "notes"
VOCABULARY_FILENAME = "vocabulary"
HASH_FILENAME = "dataset_hash"
RESULTS_DIR = "results"
SEQUENCE_LENGTH = 100
VALIDATION_SPLIT = 0.2

"""
changing this value requires refactoring

predict.py -> loop inside generate_notes() [getting prediction]
data_preparation.py -> loop inside prepare_sequences_for_training() [out sequences]
"""
NUM_NOTES_TO_PREDICT = 1


def clean_data_and_checkpoints():
    shutil.rmtree(DATA_DIR)
    shutil.rmtree(CHECKPOINTS_DIR)


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
    print(f"Parsing {file}")

    midi = converter.parse(file)
    notes = []
    try:
        # file has instrument parts
        instrument_stream = instrument.partitionByInstrument(midi)
        notes_to_parse = instrument_stream.parts[0].recurse()
    except:
        # file has notes in a flat structure
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(element.name)
        elif isinstance(element, chord.Chord):
            notes.append(".".join(str(n) for n in element.normalOrder))

    return notes


def get_notes_from_dataset():
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


def prepare_sequences_for_training(notes, vocab, vocab_size, batch_size):
    training_split = 1 - VALIDATION_SPLIT
    dataset_split = math.ceil(training_split * len(notes))
    training_sequence = NotesSequence(
        notes[:dataset_split],
        batch_size,
        SEQUENCE_LENGTH,
        vocab,
        vocab_size,
        NUM_NOTES_TO_PREDICT,
    )
    validation_sequence = NotesSequence(
        notes[dataset_split:],
        batch_size,
        SEQUENCE_LENGTH,
        vocab,
        vocab_size,
        NUM_NOTES_TO_PREDICT,
    )

    return training_sequence, validation_sequence


def prepare_sequence_for_prediction(notes, vocab):
    if len(notes) < SEQUENCE_LENGTH:
        print(
            f"File is to short. Min length: {SEQUENCE_LENGTH} sounds, provided: {len(notes)}."
        )
        sys.exit(1)

    sequence_in = notes[:SEQUENCE_LENGTH]
    network_input = [get_best_representation(vocab, sound) for sound in sequence_in]

    return network_input


def get_best_representation(vocab, pattern):
    """assumption: all 12 single notes are present in vocabulary"""
    if pattern in vocab.keys():
        return vocab[pattern]

    chord_sounds = [int(sound) for sound in pattern.split(".")]
    unknown_chord = chord.Chord(chord_sounds)
    root_note = unknown_chord.root()
    print(f"*** Mapping {unknown_chord} to {root_note} ***")
    return vocab[root_note.name]


def save_midi_file(prediction_output):
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
    midi_stream.write("midi", fp=f"{RESULTS_DIR}/{output_name}.mid")
