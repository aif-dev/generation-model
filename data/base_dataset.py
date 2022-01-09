import os
import pickle
import checksumdir
from abc import ABC, abstractmethod

BUFFER_SIZE = 1000


class BaseDataset(ABC):
    """Base interface for different ways of data handling"""

    def __init__(self,
                 data_path: str,
                 rundir: str,
                 input_shape: int,
                 vocab_name: str,
                 notes_name: str,
                 batch_size: int = None):
        self.data_path = data_path
        self.rundir = rundir
        self.input_shape = input_shape
        self.vocab_name = vocab_name
        self.notes_name = notes_name
        self.batch_size = batch_size

    @abstractmethod
    def create(self):
        raise NotImplementedError()

    @abstractmethod
    def generate_midi(self, prediction_output):
        raise NotImplementedError()

    def save_data_hash(self, hash_value):
        if not os.path.isdir(self.rundir):
            os.makedirs(self.rundir)

        hash_file_path = os.path.join(self.rundir, self.notes_name + "_hash")
        with open(hash_file_path, "wb") as hash_file:
            pickle.dump(hash_value, hash_file)

    def is_data_changed(self, dataset_path):
        current_hash = checksumdir.dirhash(dataset_path)

        hash_file_path = os.path.join(self.rundir, self.notes_name + "_hash")
        if not os.path.exists(hash_file_path):
            self.save_data_hash(current_hash)
            return True

        with open(hash_file_path, "rb") as hash_file:
            previous_hash = pickle.load(hash_file)

        if previous_hash != current_hash:
            self.save_data_hash(current_hash)
            return True

        return False
