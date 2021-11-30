import os
import datetime
import tensorflow as tf
import argparse
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from data.data_preparation import (
    get_notes_from_dataset,
    prepare_sequences_for_training,
    create_vocabulary_for_training,
    clean_data_and_checkpoints,
)
from data.data_preparation import SEQUENCE_LENGTH, NUM_NOTES_TO_PREDICT, CHECKPOINTS_DIR
from models.factory import get_model


LOG_DIR = "logs/"


def get_latest_checkpoint():
    if not os.path.isdir(CHECKPOINTS_DIR):
        os.makedirs(CHECKPOINTS_DIR)
        return None

    checkpoints = [os.path.join(CHECKPOINTS_DIR, name) for name in os.listdir(CHECKPOINTS_DIR)]
    if checkpoints:
        return max(checkpoints, key=os.path.getctime)
    else:
        return None


def train_network(model_type, batch_size, epochs):
    notes = get_notes_from_dataset()
    vocab = create_vocabulary_for_training(notes)
    vocab_size = len(vocab)
    input_shape = (SEQUENCE_LENGTH, NUM_NOTES_TO_PREDICT)

    training_sequence, validation_sequence = prepare_sequences_for_training(notes, vocab, vocab_size, batch_size)

    latest_checkpoint = get_latest_checkpoint()

    if latest_checkpoint:
        print(f"*** Restoring from the lastest checkpoint: {latest_checkpoint} ***")
        model = load_model(latest_checkpoint)
    else:
        model = get_model(model_type, input_shape, output_shape=vocab_size)

    train(model, training_sequence, validation_sequence, epochs)


def train(model, training_sequence, validation_sequence, epochs):
    filepath = os.path.join(CHECKPOINTS_DIR, "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5")
    model_checkpoint = ModelCheckpoint(filepath, monitor="val_acc", verbose=0, save_best_only=True, mode="max")

    logdir = LOG_DIR + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=logdir)

    callbacks_list = [model_checkpoint, tensorboard]

    model.fit(
        x=training_sequence,
        validation_data=validation_sequence,
        epochs=epochs,
        callbacks=callbacks_list,
        shuffle=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Launch model training.")
    parser.add_argument("--model", help="name of the model to train", type=str, required=False)
    parser.add_argument("--batch", help="batch size", type=int, required=False, default=64)
    parser.add_argument("--epochs", help="training epochs", type=int, required=False, default=100)
    parser.add_argument("--clean", help="clean /runs", action="store_true", required=False)

    return parser.parse_args()


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    args = parse_args()

    if args.clean:
        clean_data_and_checkpoints()
    if args.model:
        train_network(args.model, args.batch, args.epochs)
