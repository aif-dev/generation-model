import sys
from ..data import pianoroll_dataset, factory
import glob
import os
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
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

    def pianoroll_preprocessing(dataset = "pianoroll", data_path='data\datasets\pianoroll', 
                                    rundir = 'pianoGAN', input_shape=(80,60), avg=486):
        # all pianorolls of shape (88, 486)
        dataset = factory.get_dataset("pianoroll", data_path, rundir, input_shape)
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

    def define_discriminator():
        model = Sequential()

        model.add(Conv1D(64, 3, padding='same', input_shape=(88,486)))
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
        model.compile(loss='mae', optimizer=opt, metrics=['Accuracy'])
        print(model.summary())
        return model

    def define_generator(latent_dim):
        start_height = 88 // 8
        start_width = 486 // 8
        
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
        model.compile(loss=loss, optimizer=optimizer, metrics=['Accuracy'])
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

    def save_models(save_path, generator, discriminator, gan, step):
        save_generator_path = f'{save_path}/generator_models'
        save_discriminator_path = f'{save_path}/discriminator_models'
        save_gan_path = f'{save_path}/gan_models'
        if not os.path.exists(save_generator_path):
            os.makedirs(save_generator_path)
        if not os.path.exists(save_discriminator_path):
            os.makedirs(save_discriminator_path)
        if not os.path.exists(save_gan_path):
            os.makedirs(save_gan_path)
        generator.save(save_generator_path + f'/generator_model' + str(step) + '.h5')
        discriminator.save(save_discriminator_path + f'/discriminator_model' + str(step)+'.h5')
        gan.save(save_gan_path + f'/gan_model' +str(step) + '.h5')

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
            disc_accuracy = 0.0
            for disc_batch in range(discriminator_batches):
                X_real, y_real = PianorollGAN.generate_real_samples(dataset, half_batch)
                disc_data_real =  discriminator.train_on_batch(X_real, y_real)
                disc_loss_real += disc_data_real[0]
                X_fake, y_fake = PianorollGAN.generate_fake_samples(generator, latent_dim, half_batch)
                disc_data_fake = discriminator.train_on_batch(X_fake, y_fake)
                disc_loss_fake += disc_data_fake[0]
            disc_loss_real /= discriminator_batches
            disc_loss_fake /= discriminator_batches
            disc_accuracy = (disc_data_real[1] + disc_data_fake[1])/2
            X_gan = PianorollGAN.generate_latent_points(latent_dim, n_batch)
            y_gan = np.ones((n_batch, 1)) * real_samples_multiplier
            g_data = gan_model.train_on_batch(X_gan, y_gan)
            g_loss= g_data[0]
            
            history['discriminator_real_loss'].append(disc_loss_real)
            history['discriminator_fake_loss'].append(disc_loss_fake)
            history['generator_loss'].append(g_loss)
            epoch = step // batch_per_epoch+1
            if step%batch_per_epoch==0:
                print('epoch: %d, discriminator_real_loss=%.3f, discriminator_fake_loss=%.3f, generator_loss=%.3f \n discriminator_accuracy = %.3f, GAN_accuracy = %.3f' % (epoch, disc_loss_real, disc_loss_fake, g_loss, disc_accuracy, g_data[1]))
            if step%save_step==0:
                pianoroll.generate_midi(pianoroll, X_fake[0]+0.001, save_path)
                PianorollGAN.save_models(save_path, generator, discriminator, gan_model, step)
        return history, gan_model, generator, discriminator

def sample_run(data_path='data\datasets\pianoroll', rundir='pianoGAN', output_length=486,   loss='binary_crossentropy', optimizer=None,latent_dim=16, real_samples_multiplier=1.0, fake_samples_multiplier=0.0, discriminator_batches=32, n_epochs=100, n_batch=128, save_step=100, save_path='saved_pianorolls'):
    pianoroll_gan = PianorollGAN()
    dataset, notes = pianoroll_gan.pianoroll_preprocessing(data_path=data_path, rundir=rundir, avg=output_length)
    dis = PianorollGAN.define_discriminator()
    gen = PianorollGAN.define_generator(latent_dim)
    gan = PianorollGAN.define_gan(gen, dis, loss=loss, optimizer=optimizer)
    hist, final_gan, final_generator, final_discriminator = PianorollGAN.train(gen, dis, gan, notes, latent_dim, dataset,
    real_samples_multiplier=real_samples_multiplier, fake_samples_multiplier=fake_samples_multiplier, discriminator_batches=discriminator_batches, 
    n_epochs=n_epochs, n_batch=n_batch, save_step=save_step, save_path=save_path)
    pianoroll_gan.save_models(save_path, final_generator, final_discriminator, final_gan, 'final')

    return hist, final_gan