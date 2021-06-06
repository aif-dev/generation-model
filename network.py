from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from data_preparation import SEQUENCE_LENGTH, NUM_NOTES_TO_PREDICT


def create_network(vocab_size, weights_filename=None):
    lstm_units = 128
    dense_units = 256
    dropout_rate = 0.3
    model = Sequential()
    model.add(
        LSTM(
            lstm_units,
            input_shape=(SEQUENCE_LENGTH, NUM_NOTES_TO_PREDICT),
            return_sequences=True,
        )
    )
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_units, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_units, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_units))
    model.add(Dense(dense_units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(vocab_size))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["acc"])

    if weights_filename:
        print(f"*** Loading weights from {weights_filename} ***")
        model.load_weights(weights_filename)

    model.summary()

    return model
