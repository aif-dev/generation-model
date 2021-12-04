import argparse
import tensorflow as tf
from pathlib import Path

from utils.checkpoints import get_best_checkpoint
from models.factory import get_model
from data.factory import get_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Launch model prediction.")
    parser.add_argument("--model", help="name of the model to run prediction on", type=str, required=True)
    parser.add_argument("--dataset", help="name of the dataset to process data", type=str, required=True)
    parser.add_argument("--data", help="path to test data", type=str, required=True)
    parser.add_argument(
        "--rundir", help="path to dir storing run artifacts (vocab, notes...)", type=Path, default="proj/data"
    )
    parser.add_argument("--result_dir", help="path to dir storing predictions", type=Path, default="./results")
    parser.add_argument("--checkpoint", help="path to checkpoint dir", type=Path, default="proj/checkpoints")
    parser.add_argument("--sequence_length", help="input sequence length", type=int, default=100)
    parser.add_argument("--num_notes_predict", help="number of notes to generate", type=int, default=300)

    return parser.parse_args()


def main():
    args = parse_args()

    input_shape = (args.sequence_length, 1)
    dataset = get_dataset(args.dataset, args.data, args.rundir, input_shape)

    test_sequence = dataset.create(mode="test")
    best_checkpoint = get_best_checkpoint(args.checkpoint)
    model = get_model(args.model, input_shape, (1, dataset.vocab_size), best_checkpoint)

    prediction = model.predict(test_sequence, notes_to_gen=args.sequence_length, verbose=0)

    dataset.generate_midi(prediction, args.result_dir)


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    main()
