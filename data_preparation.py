import glob
import os
import datetime
import pickle
import checksumdir
import numpy as np
from music21 import converter, instrument, stream, note, chord
from keras.utils import np_utils

MIDI_SONGS_DIR = "midi_songs"
DATA_DIR = "data"
NOTES_FILENAME = "notes"
HASH_FILENAME = "dataset_hash"
RESULTS_DIR = "results"


def save_data_hash(hash):
    if not os.path.isdir(DATA_DIR):
        os.mkdir(DATA_DIR)

    hash_file_path = os.path.join(DATA_DIR, HASH_FILENAME)
    with open(hash_file_path, "wb") as hash_file:
        pickle.dump(hash, hash_file)


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


def get_notes_from_dataset():
    """Get all the notes and chords from the midi files in the ./midi_songs directory"""

    notes_path = os.path.join(DATA_DIR, NOTES_FILENAME)
    notes = []
    if is_data_changed():
        try:
            for file in glob.glob(f"{MIDI_SONGS_DIR}/*.mid"):
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

            with open(notes_path, "wb") as notes_path:
                pickle.dump(notes, notes_path)

        except:
            hash_file_path = os.path.join(DATA_DIR, HASH_FILENAME)
            os.remove(hash_file_path)
            print(f"Removed the hash file")
            exit(1)

    else:
        with open(notes_path, "rb") as notes_path:
            notes = pickle.load(notes_path)

    return notes


def prepare_sequences_for_training(notes, n_vocab):
    """Prepare the sequences used by the Neural Network"""
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i : i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    unnormalized_network_input = np.reshape(
        network_input, (n_patterns, sequence_length, 1)
    )
    normalized_network_input = unnormalized_network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return (normalized_network_input, network_output)


def prepare_sequences_for_prediction(notes, pitchnames, n_vocab):
    """Prepare the sequences used by the Neural Network"""
    # map between notes and integers and back
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 100
    network_input = []
    output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i : i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    unnormalized_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    normalized_input = unnormalized_input / float(n_vocab)

    return (network_input, normalized_input)


def generate_notes(model, network_input, pitchnames, n_vocab):
    """Generate notes from the neural network based on a sequence of notes"""
    # pick a random sequence from the input as a starting point for the prediction
    start = np.random.randint(0, len(network_input) - 1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    # generate 500 notes
    for _ in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1 : len(pattern)]

    return prediction_output


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
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write("midi", fp=f"{RESULTS_DIR}/output-{datetime.datetime.now()}.mid")
