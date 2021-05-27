from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import BatchNormalization as BatchNorm
from keras.layers import Activation
from data_preparation import SEQUENCE_LENGTH, NUM_NOTES_TO_PREDICT


def create_network(vocab_size, weights_filename=None):
    model = Sequential()
    model.add(
        LSTM(
            512,
            input_shape=(SEQUENCE_LENGTH, NUM_NOTES_TO_PREDICT),
            return_sequences=True,
        )
    )
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(512))
    model.add(BatchNorm())
    # model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(BatchNorm())
    # model.add(Dropout(0.3))
    model.add(Dense(vocab_size))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

    if weights_filename:
        print(f"*** Loading weights from {weights_filename} ***")
        model.load_weights(weights_filename)

    return model
