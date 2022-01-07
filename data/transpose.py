"""Transpose midi files scores to c major"""

import glob
import music21
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count



def transpose_file(file):

    majors = dict([("A-", 4),("G#", 4),("A", 3),("A#", 2),("B-", 2),("B", 1),("C", 0),("C#", -1),("D-", -1),("D", -2),("D#", -3),("E-", -3),("E", -4),("F", -5),("F#", 6),("G-", 6),("G", 5)])
    minors = dict([("G#", 1), ("A-", 1),("A", 0),("A#", -1),("B-", -1),("B", -2),("C", -3),("C#", -4),("D-", -4),("D", -5),("D#", 6),("E-", 6),("E", 5),("F", 4),("F#", 3),("G-", 3),("G", 2)])


    score = music21.converter.parse(file)
    key = score.analyze('key')

    if key.mode == "major":
        halfSteps = majors[key.tonic.name]
    
    elif key.mode == "minor":
        halfSteps = minors[key.tonic.name]

    newscore = score.transpose(halfSteps)
    key = newscore.analyze('key')


    newscore.write('midi', file)
    print(file)

def transpose_dir(dir_path):
    with Pool(cpu_count() - 1) as pool:
        pool.map(transpose_file, glob.glob(f"{dir_path}/*.mid"))

def parse_args():
    parser = argparse.ArgumentParser(description="Transpose dataset to c major")
    parser.add_argument("--data", help="path to training data", type=Path, default="../datasets/maestro-v3.0.0")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    transpose_dir(args.data)

    