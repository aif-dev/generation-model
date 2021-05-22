""" This module generates notes for a midi file using the
    trained neural network """
import os
import sys
import pickle
import tensorflow as tf
from network import create_network
from data_preparation import (
    save_midi_file,
    prepare_sequences_for_prediction,
    generate_notes,
)


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


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    generate_music()
