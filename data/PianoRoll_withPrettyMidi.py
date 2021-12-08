import glob
import pathlib

import pandas as pd
import pretty_midi


def get_Pianoroll_PM(midi_PM):
    midi_list = []
    for instrument in midi_PM.instruments:
        for note in instrument.notes:
            start = note.start
            end = note.end
            pitch = note.pitch
            velocity = note.velocity
            midi_list.append([start, end, pitch, velocity, instrument.name])
            midi_list = sorted(midi_list, key=lambda x: (x[0], x[2]))
    return midi_list


# Testing  sample files
data_dir = pathlib.Path('/Volumes/Media/Midi_Test')
filenames = glob.glob(str(data_dir / '*.mid*'))  # list of midi files

for fname in filenames:
    print(fname)
    midi_data = pretty_midi.PrettyMIDI(fname)
    midi_list_pm = get_Pianoroll_PM(midi_data)
# To display Pianoroll as data frame
    df = pd.DataFrame(midi_list_pm, columns=['Start', 'End', 'Pitch', 'Velocity', 'Instrument'])
    print(df)
