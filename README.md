# Classical Piano Composer

This project allows you to train a neural network to generate midi music files that make use of a single instrument

## Requirements

```
conda env create
```
## How to run

### To download datasets run scripts from :
```
/data/scripts
```
e.g. download maestro dataset:

```
sh ./data/scripts/get_maestro.sh
```

### To train the network run:

```
python train.py --model <model> --dataset <dataset> --data <data>
```
e.g.
```
python train.py --model lstm --dataset vocab --data ../datasets/maestro-v3.0.0
```

to see more options
```
python train.py -h
```

The network will use every midi file in directory provided in data_preparation.py **MIDI_SONGS_DIR** to train the network. The midi files should only contain a single instrument to get the most out of the training.

**NOTE**: You can stop the process at any point in time and the weights from the latest completed epoch will be available for text generation purposes.

### To start ngrok server with Tensorboard:

```
python tensorboard.py -h
```

### To generate music run:

```
python predict.py --model <model> --dataset <dataset> --data <data>
```
e.g.
```
python predict.py --model lstm --dataset vocab --data ../datasets/test
```

to see more options
```
python predict.py -h
```