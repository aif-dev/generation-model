import os
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from math import log, ceil, floor, sqrt
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import norm
from data_preparation import get_notes_from_dataset, create_vocabulary_for_training
from train import DATASET_PERCENT


print("Loading data...")
notes = get_notes_from_dataset()

print("Mapping data...")
notes = [hash(tuple(note)) for note in notes[: int(len(notes) * DATASET_PERCENT)]]

print("Counting occurances...")
notes_counter = Counter(notes)

print("Rearranging occurances...")
counter_size = len(notes_counter)
least_common = notes_counter.most_common(counter_size)
least_common.reverse()
occurances = [0] * counter_size

half_len = ceil(counter_size / 2)
for i in range(half_len):
    if counter_size % 2 == 1 and i == half_len - 1:
        occurances[i] = least_common[2 * i][1]
    else:
        occurances[i] = least_common[2 * i][1]
        occurances[-i - 1] = least_common[2 * i + 1][1]

log_occurances = [log(occurance, 2) for occurance in occurances]

print("Plotting...")
fig, axes = plt.subplots(2, 1)
fig.canvas.manager.set_window_title("Maestro")
fig.suptitle("MIDI chords/single notes from the Maestro dataset")

axes[0].set_title("Chords/single notes encoded as 88x1 matrices")
axes[0].bar([i for i in range(len(occurances))], log_occurances)
axes[0].set_ylabel("log(occurances, 2)")
axes[0].set_xlabel("matrix index")
axes[0].set_xlim([0, len(notes_counter)])

df_pitches = pd.DataFrame(occurances, columns=["occurances"])
axes[1].set_title("qqplot (normal distribution)")
qqplot(df_pitches["occurances"], line="s", ax=axes[1])

plt.show()
