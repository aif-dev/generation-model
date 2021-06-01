import math
from keras.models import Sequential
from keras.layers import (
    Dense,
    Dropout,
    LSTM,
    BatchNormalization as BatchNorm,
    Activation,
    GaussianNoise,
)
from keras.initializers import Constant
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from data_preparation import SEQUENCE_LENGTH, NOTE_MATRIX_SIZE
from keras.backend import mean, binary_crossentropy


def hamming(y_true, y_pred):
    return mean(y_true * (1 - y_pred) + (1 - y_true) * y_pred)


def create_network(weights_filename=None):
    model = Sequential()
    model.add(
        LSTM(
            256,
            input_shape=(SEQUENCE_LENGTH, NOTE_MATRIX_SIZE),
            return_sequences=True,
        )
    )
    model.add(GaussianNoise(0.075))
    model.add(LSTM(256, return_sequences=True))
    model.add(GaussianNoise(0.075))
    model.add(LSTM(256))
    model.add(BatchNorm())
    model.add(Activation("relu"))
    model.add(Dropout(0.3))

    p = 0.01
    initial_bias = -math.log((1 - p) / p)
    model.add(Dense(NOTE_MATRIX_SIZE, bias_initializer=Constant(initial_bias)))

    model.add(Activation("sigmoid"))
    loss = SigmoidFocalCrossEntropy(alpha=1, gamma=8.0)
    model.compile(loss=loss, optimizer="rmsprop", metrics=[hamming, "acc"])

    if weights_filename:
        print(f"*** Loading weights from {weights_filename} ***")
        model.load_weights(weights_filename)

    return model
