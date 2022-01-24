import os
import datetime
import tensorflow as tf
import argparse
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard
from models.factory import get_model
from data.factory import get_dataset
from pathlib import Path
from utils import checkpoints

LOG_DIR = "logs/"


def train(model, training_sequence, validation_sequence, epochs, checkpoint_dir):
    filepath = os.path.join(checkpoint_dir, "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5")
    model_checkpoint = ModelCheckpoint(filepath, monitor="val_acc", verbose=0, save_best_only=True, mode="max")

    logdir = LOG_DIR + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=logdir)

    callbacks_list = [model_checkpoint, tensorboard]

    model.fit(
        x=training_sequence,
        validation_data=validation_sequence,
        epochs=epochs,
        callbacks=callbacks_list,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Launch model training")
    parser.add_argument("--model", help="name of the model", type=str, required=True)
    parser.add_argument("--dataset", help="name of the dataset", type=str, required=True)
    parser.add_argument("--batch", help="batch size", type=int, default=64)
    parser.add_argument("--epochs", help="training epochs", type=int, default=100)
    parser.add_argument("--clean", help="clean run artifacts", action="store_true", required=False)
    parser.add_argument("--data", help="path to training data", type=Path, default="../datasets/maestro-v3.0.0")
    parser.add_argument("--checkpoint", help="path to checkpoint dir", type=Path, default="proj/checkpoints")
    parser.add_argument(
        "--rundir", help="path to dir storing run artifacts (vocab, notes...)", type=Path, default="proj/data"
    )
    parser.add_argument("--sequence_length", help="input sequence length", type=int, default=100)
    parser.add_argument("--num_notes_predict", help="output sequence length", type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.clean:
        checkpoints.clean_data_and_checkpoints(args.rundir, args.checkpoint)

    dataset = get_dataset(
        args.dataset, args.data, args.rundir, (args.sequence_length, args.num_notes_predict), args.batch
    )
    training_sequence, validation_sequence = dataset.create()

    latest_checkpoint = checkpoints.get_latest_checkpoint(args.checkpoint)

    input_shape = args.sequence_length

    if latest_checkpoint:
        print(f"*** Restoring from the lastest checkpoint: {latest_checkpoint} ***")
        model = load_model(latest_checkpoint)
    else:
        output_shape = args.num_notes_predict
        model = get_model(args.model, input_shape, output_shape)

    train(model, training_sequence, validation_sequence, args.epochs, args.checkpoint)


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    main()
