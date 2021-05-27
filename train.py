import os
import datetime
import getopt
import sys
import tensorflow as tf
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from network import create_network
from data_preparation import (
    get_notes_from_dataset,
    prepare_sequences_for_training,
    create_vocabulary_for_training,
)


LOG_DIR = "logs/"


def get_latest_checkpoint():
    if not os.path.isdir("checkpoints"):
        os.mkdir("checkpoints")
        return None

    checkpoints = ["checkpoints/" + name for name in os.listdir("checkpoints/")]
    if checkpoints:
        return max(checkpoints, key=os.path.getctime)
    else:
        return None


def train_network():
    notes = get_notes_from_dataset()
    vocab = create_vocabulary_for_training(notes)
    vocab_size = len(vocab)

    network_input, network_output = prepare_sequences_for_training(
        notes, vocab, vocab_size
    )

    latest_checkpoint = get_latest_checkpoint()

    if latest_checkpoint:
        print(f"*** Restoring from the lastest checkpoint: {latest_checkpoint} ***")
        model = load_model(latest_checkpoint)
    else:
        model = create_network(vocab_size)

    train(model, network_input, network_output)


def train(model, network_input, network_output):
    filepath = "checkpoints/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    model_checkpoint = ModelCheckpoint(
        filepath, monitor="loss", verbose=0, save_best_only=True, mode="min"
    )

    early_stopping = EarlyStopping(monitor="val_loss", patience=3)

    logdir = LOG_DIR + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=logdir)

    callbacks_list = [model_checkpoint, early_stopping, tensorboard]

    model.fit(
        network_input,
        network_output,
        validation_split=0.2,
        epochs=200,
        batch_size=128,
        callbacks=callbacks_list,
    )


def parse_cli_args():
    usage_str = (
        f"Usage: {sys.argv[0]} [-h] [-c | --clean (clean data/ and checkpoints/)]"
    )

    try:
        opts, _ = getopt.getopt(sys.argv[1:], "hc", ["clean"])
    except getopt.GetoptError:
        print(usage_str)
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print(usage_str)
            sys.exit(0)
        elif opt in ["-c", "--clean"]:
            is_file_present = True
            file = arg


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    train_network()
