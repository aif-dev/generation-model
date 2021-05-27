import shutil
import os
from zipfile import ZipFile
import wget


DATASET_PATH = "./midi_songs"
MAESTRO_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"


def download_data():
    filename = wget.download(MAESTRO_URL)
    with ZipFile(filename, "r") as zipf:
        zipf.extractall()
    os.remove(filename)


def move_data():
    maestro_path = "./maestro-v3.0.0"

    directories = os.listdir(maestro_path)
    for directory in directories:
        path = os.path.join(maestro_path, directory)
        if os.path.isdir(path):
            print(path)
            for file in os.listdir(path):
                new_filename = file.split(".")
                new_filename = f"{new_filename[0]}.mid"
                os.replace(os.path.join(path, file), new_filename)

    shutil.rmtree(maestro_path)


if __name__ == "__main__":
    if not os.path.isdir(DATASET_PATH):
        os.mkdir(DATASET_PATH)

    os.chdir(DATASET_PATH)
    download_data()
    move_data()
