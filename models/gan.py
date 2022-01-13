import sys
from ..data import pianoroll_dataset, factory
import glob
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Conv1DTranspose
from tensorflow.keras.layers import Activation
from tensorflow.keras import activations

class PianorollGAN(Sequential):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def pianoroll_preprocessing(dataset = "pianoroll", data_path='..\data\datasets\pianoroll', 
                                    rundir = 'pianoGAN', input_shape=(80,60), avg=486):
        # all pianorolls of shape (88, 486)
        dataset = factory.get_dataset(dataset, data_path, rundir, input_shape)
        dataset.get_pianorolls()
        zeros = np.zeros((1, 88))
        for i in range(len(dataset.notes)):
            if dataset.notes[i].shape[0] > avg:
               dataset.notes[i] = np.delete(dataset.notes[i], slice(avg, dataset.notes[i].shape[0]),0)
            else:
                diff = avg - dataset.notes[i].shape[0]
                for j in range(diff):
                    dataset.notes[i] = np.append(dataset.notes[i], zeros, axis=0) 

        notes = np.zeros((len(dataset.notes), 88, avg))
        for i in range(len(dataset.notes)):
            notes[i] = np.swapaxes(dataset.notes[i],0,1)

        return dataset, notes

    def define_discriminator(in_shape):
        model = Sequential()

        model.add(Conv1D(64, 3, padding='same', input_shape=in_shape))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv1D(128, 3, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv1D(128, 3, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv1D(256, 3, strides=2, padding='same'))   
        model.add(LeakyReLU(alpha=0.2))

        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Activation(activations.sigmoid))
        opt = Adam(lr=0.04, beta_1=0.5)
        model.compile(loss='mae', optimizer=opt)
        print(model.summary())
        return model

    def define_generator(latent_dim, output_shape):
        start_height = output_shape[0] // 8
        start_width = output_shape[1] // 8
        
        model = Sequential()

        n_nodes = 64* start_width * start_height
        model.add(Dense(n_nodes, input_dim=latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((8*start_height, 8*start_width)))

        model.add(Conv1DTranspose(128, 4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((8*start_height, 256)))
        
        model.add(Conv1DTranspose(128, 4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Conv1DTranspose(128, 4, strides=3, padding='same'))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.2))
        model.add(Conv1D(81, 3, activation='tanh', padding='same'))
        model.add(Reshape((88,486)))
        print(model.summary())
        return model

    def define_gan(g_model, d_model, loss='binary_crossentropy', optimizer=None):
        d_model.trainable = False
        model = Sequential()
        model.add(g_model)
        model.add(d_model)
        if not optimizer:
            optimizer = Adam(lr=0.04, beta_1=0.5)
        model.compile(loss=loss, optimizer=optimizer)
        model.add(Flatten())
        print(model.summary())
        return model

    def generate_real_samples(dataset, n_samples):
        ix = np.random.randint(0, len(dataset), n_samples)
        X = np.array(dataset)[ix]
        for i in range(ix.shape[0]):
            X[i]=np.asarray(X[i], dtype=np.float32)

        y = np.ones((n_samples, 1))
        return X, y

    def generate_latent_points(latent_dim, n_samples):
        x_input = np.random.randn(latent_dim * n_samples)
        x_input = x_input.reshape(n_samples, latent_dim)
        return x_input

    def generate_fake_samples(generator, latent_dim, n_samples):
        x_input = PianorollGAN.generate_latent_points(latent_dim, n_samples)
        X = generator.predict(x_input)
        y = np.ones((n_samples, 1)) 
        return X, y

    def train(generator, discriminator, gan_model, dataset, latent_dim, pianoroll,
          real_samples_multiplier=0.8, fake_samples_multiplier=0.0, discriminator_batches=2,
          n_epochs=500, n_batch=128, save_step=1000, save_path="pianoroll_samples"):

        batch_per_epoch = len(dataset)// n_batch
        half_batch = n_batch // 2
        seed = PianorollGAN.generate_latent_points(latent_dim, 87)
        n_steps = batch_per_epoch * n_epochs
        
        history = {'discriminator_real_loss': [],
                'discriminator_fake_loss': [],
                'generator_loss': []}
        for step in range(n_steps):
            epoch = step // batch_per_epoch
            disc_loss_real = 0.0
            disc_loss_fake = 0.0
            for disc_batch in range(discriminator_batches):
                X_real, y_real = PianorollGAN.generate_real_samples(dataset, half_batch)

                disc_loss_real += discriminator.train_on_batch(X_real, y_real)
                X_fake, y_fake = PianorollGAN.generate_fake_samples(generator, latent_dim, half_batch)
                disc_loss_fake += discriminator.train_on_batch(X_fake, y_fake)
            disc_loss_real /= discriminator_batches
            disc_loss_fake /= discriminator_batches
            
            X_gan = PianorollGAN.generate_latent_points(latent_dim, n_batch)
            y_gan = np.ones((n_batch, 1)) * real_samples_multiplier
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            
            history['discriminator_real_loss'].append(disc_loss_real)
            history['discriminator_fake_loss'].append(disc_loss_fake)
            history['generator_loss'].append(g_loss)
            epoch = step // batch_per_epoch+1
            if step%batch_per_epoch==0:
                print('epoch: %d, discriminator_real_loss=%.3f, discriminator_fake_loss=%.3f, generator_loss=%.3f' % (epoch, disc_loss_real, disc_loss_fake, g_loss))
            if step%save_step==0:
                pianoroll.generate_midi(X_fake[0], save_path)

        return history, gan_model, generator, discriminator

        
def sample_run(save_gan=False, save_path=None):
    pianoroll_gan = PianorollGAN()
    dataset, notes = pianoroll_gan.pianoroll_preprocessing()
    dis = pianoroll_gan.define_discriminator((88,486))
    gen = pianoroll_gan.define_generator(16, (88,486))
    gan = pianoroll_gan.define_gan(gen, dis)
    hist, final_gan, final_generator, final_discriminator = pianoroll_gan.train(gen, dis, gan, notes, 16, pianoroll=dataset)
    if(save_gan and save_path is not None):
        save_generator_path = f'{save_path}/generator_models'
        save_discriminator_path = f'{save_path}/discriminator_models'
        save_gan_path = f'{save_path}/gan_models'
        if not os.path.exists(save_generator_path):
            os.makedirs(save_generator_path)
        if not os.path.exists(save_discriminator_path):
            os.makedirs(save_discriminator_path)
        if not os.path.exists(save_gan_path):
            os.makedirs(save_gan_path)
        final_generator.save(save_generator_path + f'/generator_model.h5')
        final_discriminator.save(save_discriminator_path + f'/discriminator_model.h5')
        final_gan.save(save_gan_path + f'/gan_model.h5')

    return hist, final_gan

