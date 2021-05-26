""" This module generates notes for a midi file using the
    trained neural network """
import os
import sys
import pickle
import tensorflow as tf
import numpy as np
from network import create_network
from data_preparation import save_midi_file, prepare_sequences_for_prediction, get_notes_from_file


def get_best_weights_filename():
    checkpoints = ["checkpoints/" + name for name in os.listdir("checkpoints/")]

    if not checkpoints:
        raise Exception("Couldn't find any weights in the checkpoints/ directory.")

    lowest_loss = sys.float_info.max
    best_checkpoint = checkpoints[0]

    for checkpoint in checkpoints:
        loss = float(checkpoint.split("-")[3])

        if loss < lowest_loss:
            best_checkpoint = checkpoint

    print(f"*** Found checkpoint with the best weights: {best_checkpoint} ***")
    return best_checkpoint


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


def generate_music():
    """Generate a piano midi file"""
    # load the notes used to train the model
    with open("data/notes", "rb") as filepath:
        notes = pickle.load(filepath)

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))
    # Get all pitch names
    n_vocab = len(set(notes))

    network_input, normalized_input = prepare_sequences_for_prediction(
        notes, pitchnames, n_vocab
    )
    model = create_network(normalized_input, n_vocab, get_best_weights_filename())
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    save_midi_file(prediction_output)


def generate_music_from_file(filename):
    notes = get_notes_from_file(filename)
    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))
    # Get all pitch names
    n_vocab = len(set(notes))

    network_input, normalized_input = prepare_sequences_for_prediction(
        notes, pitchnames, n_vocab
    )
    model = create_network(normalized_input, n_vocab, get_best_weights_filename())
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    save_midi_file(prediction_output)


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    generate_music()
