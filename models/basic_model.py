from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LSTM,
    Activation,
)

def create_network(input_shape, output_shape, weights_filename=None):
    model = Sequential()
    model.add(
        LSTM(
            64,
            input_shape=input_shape,
            return_sequences=True,
        )
    )
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(Dense(128))
    model.add(Dropout(0.3))
    model.add(Dense(output_shape))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["acc"])

    model.summary()

    if weights_filename:
        print(f"*** Loading weights from {weights_filename} ***")
        model.load_weights(weights_filename)

    return model
