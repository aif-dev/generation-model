import os
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.gofplots import qqplot
from data_preparation import get_notes_from_dataset, create_vocabulary_for_training
from train import DATASET_PERCENT

notes = get_notes_from_dataset()
notes = notes[: int(len(notes) * DATASET_PERCENT)]
vocab = create_vocabulary_for_training(notes)
vocab_size = len(vocab)
mapped_notes = [vocab[note] for note in notes]

fig, axes = plt.subplots(2, 1)

fig.canvas.set_window_title("Maestro")
fig.suptitle("MIDI chords/notes from the Maestro dataset")

sorted_mapped_notes_counter = Counter(sorted(mapped_notes))
axes[0].set_title(
    "Distribution of single notes and chords (note.note.note...) in MIDI mapped to vocabulary which is sorted by the lowest MIDI value in a chord"
)
axes[0].bar(sorted_mapped_notes_counter.keys(), sorted_mapped_notes_counter.values())
axes[0].set_ylabel("n")
axes[0].set_xlabel("pattern")
axes[0].set_xlim([0, vocab_size])

df_pitches = pd.DataFrame(mapped_notes, columns=["mapped_sound"])
axes[1].set_title("qqplot (normal distribution)")
qqplot(df_pitches["mapped_sound"], line="s", ax=axes[1])

plt.show()
