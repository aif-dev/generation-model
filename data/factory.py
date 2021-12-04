from data import vocab_dataset


def get_dataset(dataset, data_path, rundir, input_shape, batch_size=None):
    if dataset == "vocab":
        return vocab_dataset.VocabDataset(
            data_path=data_path, rundir=rundir, input_shape=input_shape, batch_size=batch_size
        )
    else:
        raise ValueError(f"There is no dataset with name {dataset}")
