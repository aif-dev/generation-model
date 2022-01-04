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

class Visualizer():

    def __init__(self):
        self.filenames = glob.glob(str(pathlib.Path('./datasets/pianoroll')/'*'))
        self.notes = collections.defaultdict(list)
        self.parse_file()

    def parse_file(self):
        #do some parsing
        print("TEST1")
        for file in self.filenames:
          pm = pretty_midi.PrettyMIDI(file)
          instrument = pm.instruments[0]

          # Sort the notes by start time
          sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
          for note in sorted_notes:
            self.notes['pitch'].append(note.pitch)

        return pd.DataFrame({name: np.array(value) for name, value in self.notes.items()})

    def plot_distributions(self):
      print("TEST2")
      sns.histplot(self.notes, x="pitch")
      sns.set(rc={'figure.figsize':(100.55,8.27)})
      plt.xticks(rotation=90)
      plt.show()
      