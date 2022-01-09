#!/bin/bash
# Dataset containing midi piano rolls.
# Example usage: sh data/scripts/get_piano.sh
# parent
# ├── generation-model
# └── datasets
#     └── pianoroll  ← downloads here

d='../datasets/pianoroll'
url='http://www.pianola.co.nz/public/bulk_midi/'
f='Raspin_MIDI_Rollscans_from_pianola.co.nz.zip'

echo 'Downloading' $url$f
curl -L $url$f -o $f && mkdir -p $d && unzip -q $f -d $d && rm $f

python - << EOF
import os
piano_path = "../datasets/pianoroll"
directories = os.listdir(piano_path)
for directory in directories:
    path = os.path.join(piano_path, directory)
    if os.path.isdir(path):
        for file in os.listdir(path):
            new_filename = file.split(".")
            new_filename = f"{new_filename[0]}.mid"
            os.replace(os.path.join(path, file), os.path.join(piano_path, new_filename))
EOF