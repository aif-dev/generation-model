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
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from data_preparation import SEQUENCE_LENGTH, NOTE_MATRIX_SIZE


def create_network(weights_filename=None):
    model = Sequential()
    model.add(
        LSTM(
            128,
            input_shape=(SEQUENCE_LENGTH, NOTE_MATRIX_SIZE),
            return_sequences=True,
        )
    )
    model.add(GaussianNoise(0.075))
    model.add(LSTM(128, return_sequences=True))
    model.add(GaussianNoise(0.075))
    model.add(LSTM(128))
    model.add(BatchNorm())
    model.add(Activation("relu"))
    model.add(Dropout(0.3))

    # initally, positive samples will be extremely incorrect therefore
    # they will receive high weights
    p = 0.1
    initial_bias = -math.log((1 - p) / p)
    model.add(Dense(NOTE_MATRIX_SIZE, bias_initializer=Constant(initial_bias)))

    model.add(Activation("sigmoid"))
    # alpha - controls balance between positive and negative labeled samples
    #         alpha for positive and 1-alpha for negative
    #         positive examples are usually minorities
    # gamma - controls how much attention goes to misclassified examples
    #         less loss will be propagated from easy examples
    loss = SigmoidFocalCrossEntropy(alpha=0.05, gamma=2.0)
    model.compile(loss=loss, optimizer="rmsprop", metrics=["acc"])

    if weights_filename:
        print(f"*** Loading weights from {weights_filename} ***")
        model.load_weights(weights_filename)

    return model
