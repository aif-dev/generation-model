from keras.models import Sequential
from keras.layers import (
    Dense,
    Dropout,
    LSTM,
    BatchNormalization as BatchNorm,
    Activation,
    GRU,
    Embedding,
    Flatten,
    TimeDistributed,
    Reshape,
    Input,
    GaussianNoise,
)
from keras.optimizers import RMSprop, Adam, SGD
from keras.regularizers import l1_l2
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from data_preparation import SEQUENCE_LENGTH, NUM_NOTES_TO_PREDICT


def create_network(vocab_size, weights_filename=None):
    # our
    # val_loss ~= 1800
    #
    # model = Sequential()
    # model.add(
    #     LSTM(
    #         512,
    #         input_shape=(SEQUENCE_LENGTH, NUM_NOTES_TO_PREDICT),
    #         return_sequences=True,
    #     )
    # )
    # model.add(LSTM(512, return_sequences=True))
    # model.add(LSTM(512))
    # model.add(BatchNorm())
    # model.add(Dropout(0.3))
    # model.add(Dense(256))
    # model.add(Activation("relu"))
    # model.add(BatchNorm())
    # model.add(Dropout(0.3))
    # model.add(Dense(vocab_size))
    # model.add(Activation("softmax"))
    # model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

    # https://github.com/Skuldur/Classical-Piano-Composer/pull/21
    # val_loss ~= 5
    #
    # lstm_units = 512
    # dropout_rate = 0.3
    # model = Sequential()
    # model.add(
    #     LSTM(
    #         lstm_units,
    #         input_shape=(SEQUENCE_LENGTH, NUM_NOTES_TO_PREDICT),
    #         return_sequences=True,
    #     )
    # )
    # model.add(Dropout(dropout_rate))
    # model.add(LSTM(lstm_units, return_sequences=True))
    # model.add(Dropout(dropout_rate))
    # model.add(LSTM(lstm_units))
    # model.add(BatchNorm())
    # model.add(Dense(vocab_size))
    # model.add(Activation("softmax"))
    # model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

    # https://arxiv.org/pdf/2006.09838v1.pdf
    # val_loss ~= 3.6
    #
    # optimizer = RMSprop(
    #     learning_rate=0.0001
    # )  # after decresing lr val_loss starts to oscilate around 3.6085
    #
    # dropout_rate = 0.3
    # lstm_units = 256
    # model = Sequential()
    # model.add(
    #     LSTM(
    #         lstm_units,
    #         input_shape=(SEQUENCE_LENGTH, NUM_NOTES_TO_PREDICT),
    #         return_sequences=True,
    #     )
    # )
    # model.add(Dense(vocab_size))
    # model.add(Dropout(dropout_rate))
    # model.add(LSTM(lstm_units, return_sequences=True))
    # model.add(Dropout(dropout_rate))
    # model.add(LSTM(lstm_units))
    # model.add(Dense(256))
    # model.add(Dropout(dropout_rate))
    # model.add(Dense(vocab_size))
    # model.add(Activation("softmax"))
    # model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

    # custom
    # val_loss ~=
    #
    model = Sequential()
    model.add(
        LSTM(
            256,
            input_shape=(SEQUENCE_LENGTH, NUM_NOTES_TO_PREDICT),
            return_sequences=True,
        )
    )
    model.add(GaussianNoise(0.075))
    model.add(LSTM(256))
    model.add(Dropout(0.3))
    model.add(Dense(vocab_size))
    model.add(Activation("softmax"))
    optimizer = RMSprop(clipvalue=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["acc"])

    # https://www.tandfonline.com/doi/full/10.1080/25765299.2019.1649972
    # val_loss ~= 3.6
    #
    # model = Sequential()
    # model.add(
    #     LSTM(
    #         512,
    #         input_shape=(SEQUENCE_LENGTH, NUM_NOTES_TO_PREDICT),
    #         return_sequences=True,
    #     )
    # )
    # model.add(Dropout(0.6))
    # model.add(LSTM(512, return_sequences=True))
    # model.add(Dropout(0.6))
    # model.add(LSTM(512, return_sequences=True))
    # model.add(Dropout(0.6))
    # model.add(LSTM(512))
    # model.add(Dropout(0.6))
    # model.add(Dense(vocab_size))
    # model.add(Dropout(0.6))
    # model.add(Dense(256))
    # model.add(Dropout(0.6))
    # model.add(Dense(vocab_size))
    # model.add(Activation("softmax"))
    # model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

    # https://github.com/subpath/Keras_music_gereration/blob/master/Music%20gerenation%20with%20Keras%20and%20TF.ipynb
    # https://stackoverflow.com/questions/58382732/very-low-accuracy-with-lstm
    # val_loss ~=
    #
    # model = Sequential()
    # # model.add(
    # #     Embedding(input_dim=vocab_size, output_dim=64, input_length=SEQUENCE_LENGTH)
    # # )
    # model.add(
    #     CuDNNLSTM(
    #         256,
    #         input_shape=(SEQUENCE_LENGTH, NUM_NOTES_TO_PREDICT),
    #         return_sequences=True,
    #     )
    # )
    # # model.add(Dropout(0.6))
    # model.add(
    #     CuDNNLSTM(
    #         128,
    #         input_shape=(SEQUENCE_LENGTH, NUM_NOTES_TO_PREDICT),
    #         return_sequences=True,
    #     )
    # )
    # # model.add(Dropout(0.6))
    # model.add(
    #     CuDNNLSTM(
    #         64,
    #         input_shape=(SEQUENCE_LENGTH, NUM_NOTES_TO_PREDICT),
    #         return_sequences=False,
    #     )
    # )
    # model.add(Dropout(0.6))
    # model.add(Dense(vocab_size))
    # model.add(Activation("softmax"))
    # optimizer = Adam(lr=0.001)
    # model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["acc"])

    # https://www.atlantis-press.com/journals/ijcis/125941516/view
    # val_loss ~= 3.6
    #
    # model = Sequential()
    # model.add(
    #     GRU(
    #         512,
    #         input_shape=(SEQUENCE_LENGTH, NUM_NOTES_TO_PREDICT),
    #         return_sequences=True,
    #     )
    # )
    # model.add(Dropout(0.3))
    # model.add(GRU(512))
    # model.add(Dropout(0.3))
    # model.add(Dense(vocab_size))
    # model.add(Activation("softmax"))
    # model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

    # custom
    # val_loss ~= 4
    #
    # dropout_rate = 0.6
    # l1_regulizer_strength = 0.005
    # l2_regulizer_strength = 0.005
    # gru_units = 128
    # # optimizer = RMSprop(learning_rate=0.0001) # decreasing lr didn't help
    # model = Sequential()
    # model.add(
    #     GRU(
    #         gru_units,
    #         input_shape=(SEQUENCE_LENGTH, NUM_NOTES_TO_PREDICT),
    #         return_sequences=True,
    #         kernel_regularizer=l1_l2(l1_regulizer_strength, l2_regulizer_strength),
    #         # recurrent_regularizer=l1_l2(l1_regulizer_strength, l2_regulizer_strength),
    #     )
    # )
    # model.add(Dropout(dropout_rate))
    # model.add(
    #     GRU(
    #         gru_units,
    #         return_sequences=True,
    #         kernel_regularizer=l1_l2(l1_regulizer_strength, l2_regulizer_strength),
    #         # recurrent_regularizer=l1_l2(l1_regulizer_strength, l2_regulizer_strength),
    #     )
    # )
    # model.add(Dropout(dropout_rate))
    # model.add(
    #     GRU(
    #         gru_units,
    #         kernel_regularizer=l1_l2(l1_regulizer_strength, l2_regulizer_strength),
    #         # recurrent_regularizer=l1_l2(l1_regulizer_strength, l2_regulizer_strength),
    #     )
    # )
    # model.add(Dropout(dropout_rate))
    # model.add(
    #     Dense(
    #         vocab_size,
    #         # kernel_regularizer=l1_l2(l1_regulizer_strength, l2_regulizer_strength),
    #     )
    # )
    # model.add(Dropout(dropout_rate))
    # model.add(Activation("softmax"))
    # model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

    # custom
    # val_loss ~= 3.6
    #
    # model = Sequential()
    # model.add(
    #     LSTM(
    #         512,
    #         input_shape=(SEQUENCE_LENGTH, NUM_NOTES_TO_PREDICT),
    #     )
    # )
    # model.add(Dense(vocab_size))
    # model.add(Activation("softmax"))
    # model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

    # custom
    # val_loss ~= 3.7
    #
    # model = Sequential()
    # model.add(
    #     LSTM(
    #         512,
    #         input_shape=(SEQUENCE_LENGTH, NUM_NOTES_TO_PREDICT),
    #         return_sequences=True,
    #     )
    # )
    # model.add(Dropout(0.5))
    # model.add(LSTM(512))
    # model.add(Dropout(0.5))
    # model.add(Dense(vocab_size))
    # model.add(Activation("softmax"))
    # model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

    # https://www.researchgate.net/publication/342444314_Automatic_Music_Generator_Using_Recurrent_Neural_Network
    # val_los ~=
    #
    # model = Sequential()
    # model.add(
    #     LSTM(
    #         512,
    #         input_shape=(SEQUENCE_LENGTH, NUM_NOTES_TO_PREDICT),
    #         return_sequences=True,
    #     )
    # )
    # model.add(Dropout(0.1))
    # model.add(LSTM(512))
    # model.add(Dense(vocab_size))
    # model.add(Activation("softmax"))
    # model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

    # https://www.researchgate.net/publication/342444314_Automatic_Music_Generator_Using_Recurrent_Neural_Network
    # val_los ~=
    #
    # model = Sequential()
    # model.add(
    #     GRU(
    #         512,
    #         input_shape=(SEQUENCE_LENGTH, NUM_NOTES_TO_PREDICT),
    #         return_sequences=True,
    #     )
    # )
    # model.add(Dropout(0.1))
    # model.add(GRU(512))
    # model.add(Dense(vocab_size))
    # model.add(Activation("softmax"))
    # model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

    if weights_filename:
        print(f"*** Loading weights from {weights_filename} ***")
        model.load_weights(weights_filename)

    print(model.summary())

    return model
