import os
import sys
import argparse
import tensorflow as tf
import numpy as np
from models.basic_model import create_network
from data.data_preparation import (
    save_midi_file,
    prepare_sequence_for_prediction,
    load_vocabulary_from_training,
    get_notes_from_file,
)
from data.data_preparation import SEQUENCE_LENGTH, NUM_NOTES_TO_PREDICT, CHECKPOINTS_DIR


NUM_NOTES_TO_GENERATE = 300


def get_best_weights_filename():
    checkpoints = [os.path.join(CHECKPOINTS_DIR, name) for name in os.listdir(CHECKPOINTS_DIR)]

    if not checkpoints:
        raise Exception("Couldn't find any weights in the checkpoints/ directory.")

    lowest_loss = sys.float_info.max
    best_checkpoint = checkpoints[0]

    for checkpoint in checkpoints:
        loss = float(checkpoint.split("-")[3])

        if loss < lowest_loss:
            lowest_loss = loss
            best_checkpoint = checkpoint

    print(f"*** Found checkpoint with the best weights: {best_checkpoint} ***")
    return best_checkpoint


def generate_notes(model, network_input, vocab, vocab_size):
    inverted_vocab = {i: note for note, i in vocab.items()}
    sequence_in = [note_idx / float(vocab_size) for note_idx in network_input]
    prediction_output = []

    for _ in range(NUM_NOTES_TO_GENERATE):
        prediction_input = np.reshape(sequence_in, (1, SEQUENCE_LENGTH, NUM_NOTES_TO_PREDICT))

        prediction = model.predict(prediction_input, verbose=0)
        best_note_idx = np.argmax(prediction)
        best_note = inverted_vocab[best_note_idx]
        prediction_output.append(best_note)

        normalized_best_note_idx = best_note_idx / float(vocab_size)
        sequence_in.append(normalized_best_note_idx)

        # store only last 'SEQUENCE_LENGTH' elements for next prediction
        sequence_in = sequence_in[NUM_NOTES_TO_PREDICT : SEQUENCE_LENGTH + NUM_NOTES_TO_PREDICT]

    return prediction_output


def generate_music(file):
    notes = get_notes_from_file(file)
    vocab = load_vocabulary_from_training()
    vocab_size = len(vocab)

    network_input = prepare_sequence_for_prediction(notes, vocab)
    model = create_network((SEQUENCE_LENGTH, NUM_NOTES_TO_PREDICT), vocab_size, get_best_weights_filename())
    prediction_output = generate_notes(model, network_input, vocab, vocab_size)
    save_midi_file(prediction_output)


def parse_args():
    parser = argparse.ArgumentParser(description="Launch model prediction.")
    parser.add_argument("--model", help="name of the model to run prediction on", type=str, required=True)
    parser.add_argument("--file", help="file to run prediction on", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    args = parse_args()
    generate_music(args.file)
