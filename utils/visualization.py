import os
from numpy.core.records import array
import pretty_midi
import music21 as m21
import glob
import pathlib
import collections
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import seaborn as sns

data_dir = pathlib.Path('./datasets/pianoroll')
filenames = glob.glob(str(data_dir/'*'))

sample_file = "./datasets/pianoroll/Avalon.mid"

pm = pretty_midi.PrettyMIDI(sample_file)


def midi_to_notes(filenames: array) -> pd.DataFrame:
  notes = collections.defaultdict(list)
  for file in filenames:
    pm = pretty_midi.PrettyMIDI(file)
    instrument = pm.instruments[0]

    # Sort the notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    for note in sorted_notes:
      notes['pitch'].append(note.pitch)

  return pd.DataFrame({name: np.array(value) for name, value in notes.items()})


raw_notes = midi_to_notes(filenames)


def plot_distributions(notes: pd.DataFrame, drop_percentile=2.5):
  sns.histplot(notes, x="pitch")
  sns.set(rc={'figure.figsize':(100.55,8.27)})
  plt.xticks(rotation=90)
  plt.show()


plot_distributions(raw_notes)