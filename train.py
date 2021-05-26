import os
import datetime
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
    return max(checkpoints, key=os.path.getctime)


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
    modelCheckpoint = ModelCheckpoint(
        filepath, monitor="loss", verbose=0, save_best_only=True, mode="min"
    )

    earlyStopping = EarlyStopping(monitor="val_loss", patience=3)

    logdir = LOG_DIR + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorBoard = TensorBoard(log_dir=logdir)

    callbacks_list = [modelCheckpoint, earlyStopping, tensorBoard]

    model.fit(
        network_input,
        network_output,
        validation_split=0.2,
        epochs=200,
        batch_size=128,
        callbacks=callbacks_list,
    )


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    train_network()
