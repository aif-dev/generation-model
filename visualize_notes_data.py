import os
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from math import ceil, sqrt
from statsmodels.graphics.gofplots import qqplot
from data_preparation import get_notes_from_dataset, create_vocabulary_for_training
from train import DATASET_PERCENT

print("Loading data...")
notes = get_notes_from_dataset()
notes = notes[: int(len(notes) * DATASET_PERCENT)]

print("Creating vocabulary...")
vocab = create_vocabulary_for_training(notes)
vocab_size = len(vocab)

print("Mapping notes using vocabulary...")
mapped_notes = [vocab[note] for note in notes]

print("Counting occurances...")
notes_counter = Counter(mapped_notes)

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

sqrt_occurances = [sqrt(occurance) for occurance in occurances]

print("Plotting...")
fig, axes = plt.subplots(2, 1)
fig.canvas.set_window_title("Maestro")
fig.suptitle("MIDI chords/notes from the Maestro dataset")

axes[0].set_title(
    "Distribution of single notes and chords (note.note.note...) in MIDI mapped to vocabulary which is sorted by the lowest MIDI value in a chord"
)
axes[0].bar([i for i in range(len(occurances))], sqrt_occurances)
axes[0].set_ylabel("sqrt(occurances)")
axes[0].set_xlabel("chord")
axes[0].set_xlim([0, len(occurances)])

df_pitches = pd.DataFrame(occurances, columns=["occurances"])
axes[1].set_title("qqplot (normal distribution)")
qqplot(df_pitches["occurances"], line="s", ax=axes[1])

plt.show()
