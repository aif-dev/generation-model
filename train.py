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
    clean_data_and_checkpoints,
)


LOG_DIR = "logs/"
BATCH_SIZE = 64


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

    training_sequence, validation_sequence = prepare_sequences_for_training(
        notes, vocab, vocab_size, BATCH_SIZE
    )

    latest_checkpoint = get_latest_checkpoint()

    if latest_checkpoint:
        print(f"*** Restoring from the lastest checkpoint: {latest_checkpoint} ***")
        model = load_model(latest_checkpoint)
    else:
        model = create_network(vocab_size)

    train(model, training_sequence, validation_sequence)


def train(model, training_sequence, validation_sequence):
    filepath = "checkpoints/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    model_checkpoint = ModelCheckpoint(
        filepath, monitor="val_acc", verbose=0, save_best_only=True, mode="max"
    )

    logdir = LOG_DIR + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=logdir)

    callbacks_list = [model_checkpoint, tensorboard]

    model.fit(
        x=training_sequence,
        validation_data=validation_sequence,
        epochs=200,
        callbacks=callbacks_list,
        shuffle=True,
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

    for opt, _ in opts:
        if opt == "-h":
            print(usage_str)
            sys.exit(0)
        elif opt in ["-c", "--clean"]:
            clean_data_and_checkpoints()


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    parse_cli_args()
    train_network()
