""" This module prepares midi file data and feeds it to the neural
    network for training """
import os
import tensorflow as tf
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from network import create_network
from data_preparation import get_notes_from_dataset, prepare_sequences_for_training


def train_network():
    """Train a Neural Network to generate music"""
    notes = get_notes_from_dataset()

    # get amount of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences_for_training(notes, n_vocab)

    if not os.path.isdir("checkpoints"):
        os.mkdir("checkpoints")

    checkpoints = ["checkpoints/" + name for name in os.listdir("checkpoints/")]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"*** Restoring from the lastest checkpoint: {latest_checkpoint} ***")
        model = load_model(latest_checkpoint)
    else:
        model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)


def train(model, network_input, network_output):
    """train the neural network"""
    filepath = "checkpoints/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"

    modelCheckpoint = ModelCheckpoint(
        filepath, monitor="loss", verbose=0, save_best_only=True, mode="min"
    )

    earlyStopping = EarlyStopping(monitor="loss", patience=3)

    callbacks_list = [modelCheckpoint, earlyStopping]

    model.fit(
        network_input,
        network_output,
        epochs=200,
        batch_size=128,
        callbacks=callbacks_list,
    )


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    train_network()
