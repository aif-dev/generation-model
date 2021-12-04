import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LSTM,
    Activation,
)


class LstmModel(Sequential):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # self.model = self.create_network(input_shape, output_shape, weights_filename)

    def predict(self, input_sequence, notes_to_gen, normalize=True, **kwargs):
        prediction_output = []
        input_sequence = tf.expand_dims(input_sequence, 0)
        for _ in range(notes_to_gen):
            pred_logits = super().predict(input_sequence, **kwargs)

            note = tf.random.categorical(pred_logits, num_samples=1)
            note = int(tf.squeeze(note, axis=-1))

            prediction_output.append(note)

            if normalize:
                note /= self.output_shape[1]

            note = np.reshape([note], (1,1))

            input_sequence = np.delete(input_sequence, 0, axis=1)
            input_sequence = np.append(input_sequence, np.expand_dims(note, axis=0), axis=1)

        return prediction_output

def create_network(input_shape, output_shape, weights_filename=None):
    model = LstmModel()
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
    model.add(Dense(output_shape[1]))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["acc"])

    model.summary()

    if weights_filename:
        print(f"*** Loading weights from {weights_filename} ***")
        model.load_weights(weights_filename)

    return model
