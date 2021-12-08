import glob
import pathlib

import music21 as m21
import pandas as pd

def get_Pianoroll(midi):
    Pianoroll = []
    for part in midi.parts:
        instrument = part.getInstrument().instrumentName
    if instrument=="Piano":
        for note in part.flat.notes:

            if note.isChord:
                start = note.offset
                duration = note.quarterLength

                for chord_note in note.pitches:
                    pitch = chord_note.ps
                    volume = note.volume.realized
                    Pianoroll.append([start, duration, pitch, volume, instrument])

            else:
                start = note.offset
                duration = note.quarterLength
                pitch = note.pitch.ps
                volume = note.volume.realized
                Pianoroll.append([start, duration, pitch, volume, instrument])

    Pianoroll = sorted(Pianoroll, key=lambda x: (x[0], x[2]))
    return Pianoroll

# Testing  sample files
data_dir = pathlib.Path('/Volumes/Media/Midi_Test')
filenames = glob.glob(str(data_dir / '*.mid*'))  # list of midi files

for fname in filenames:
    print(fname)
    midi_data = m21.converter.parse(fname)

    PR = get_Pianoroll(midi_data)
    #To display Pianoroll as data frame
    df = pd.DataFrame(PR, columns=['Start', 'End', 'Pitch', 'Velocity', 'Instrument'])
    #Visual representation of the data frame
    # html = df.to_html(index=False, float_format='%.2f', max_rows=8)
    # ipd.HTML(html)
    print(df)
# print(f"Parsing {file}")
