#!/bin/bash
# Huge dataset containing 200h of piano music.
# Example usage: sh data/scripts/get_maestro.sh
# parent
# ├── generation-model
# └── datasets
#     └── maestro-v3.0.0  ← downloads here

d='../datasets'
url='https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/'
f='maestro-v3.0.0-midi.zip'

echo 'Downloading' $url$f
curl -L $url$f -o $f && mkdir -p '../datasets/' && unzip -q $f -d $d && rm $f

rm -- ../datasets/maestro-v3.0.0/* 2> /dev/null

python - << EOF
import os
maestro_path = "../datasets/maestro-v3.0.0"
directories = os.listdir(maestro_path)
for directory in directories:
    path = os.path.join(maestro_path, directory)
    if os.path.isdir(path):
        for file in os.listdir(path):
            new_filename = file.split(".")
            new_filename = f"{new_filename[0]}.mid"
            os.replace(os.path.join(path, file), os.path.join(maestro_path, new_filename))
EOF

rm -r -- ../datasets/maestro-v3.0.0/*/