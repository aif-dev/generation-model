import pickle
from math import sqrt
import numpy as np
from matplotlib import pyplot
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
#from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Conv1DTranspose
from tensorflow.keras.layers import Activation
from tensorflow.keras import activations

class MuseGAN():
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def define_discriminator(in_shape):
        model = Sequential()
        # normal
        
        model.add(Conv1D(64, 3, padding='same', input_shape=in_shape))
        model.add(LeakyReLU(alpha=0.2))
        # downsample
        model.add(Conv1D(128, 3, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # downsample
        model.add(Conv1D(128, 3, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # downsample
        model.add(Conv1D(256, 3, strides=2, padding='same'))   
        model.add(LeakyReLU(alpha=0.2))
        # classifier
        model.add(Flatten())
        model.add(Dropout(0.4))
        #model.add(Dense(1, activation='sigmoid'))
        model.add(Activation(activations.sigmoid))
        # compile model
        opt = Adam(lr=0.0004, beta_1=0.5)
        model.compile(loss='mae', optimizer=opt)
        print(model.summary())
        return model

    def define_generator(latent_dim, output_shape):
        start_height = output_shape[0] // 8
        start_width = output_shape[1] // 8
        
        model = Sequential()

        n_nodes = 256* start_width * start_height
        model.add(Dense(n_nodes, input_dim=latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Reshape((4*start_height, 8*start_width)))
        model.add(Reshape((16*start_height, 16*start_width)))
        
        model.add(Conv1DTranspose(128, 4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Reshape((8*start_height, 16*start_width)))
        
        model.add(Conv1DTranspose(128, 4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Reshape((16*start_height, 16*start_width)))
        
        model.add(Conv1DTranspose(128, 4, strides=3, padding='same'))
        model.add(LeakyReLU(alpha=0.2))

        # output layer
        model.add(Conv1D(5, 3, activation='tanh', padding='same'))
        model.add(Reshape((5000, 120)))
        print(model.summary())
        return model

    def define_gan(g_model, d_model, loss='binary_crossentropy', optimizer=None):
        # make weights in the discriminator not trainable
        d_model.trainable = False
        # connect them
        model = Sequential()
        # add generator
        model.add(g_model)
        # add the discriminator
        model.add(d_model)
        # compile model
        if not optimizer:
            optimizer = Adam(lr=0.0004, beta_1=0.5)
        model.compile(loss=loss, optimizer=optimizer)
        model.add(Flatten())
        print(model.summary())
        return model

    # select real samples
    def generate_real_samples(dataset, n_samples, multiplier=1.0):
        # choose random instances
        ix = np.random.randint(0, dataset.shape[0], n_samples)
        # select images
        X = dataset[ix]
        # generate class labels, -1 for 'real'
        y = np.ones((n_samples, 1)) * multiplier
        return X, y

# generate points in latent space as input for the generator
    def generate_latent_points(latent_dim, n_samples):
        # generate points in the latent space
        x_input = np.random.randn(latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(n_samples, latent_dim)
        return x_input

# use the generator to generate n fake examples, with class labels
    def generate_fake_samples(generator, latent_dim, n_samples, multiplier=1.0):
        # generate points in latent space
        x_input = MuseGAN.generate_latent_points(latent_dim, n_samples)
        # predict outputs
        X = generator.predict(x_input)
        # create class labels with 1.0 for 'fake'
        y = np.ones((n_samples, 1)) * multiplier
        return X, y

# train the generator and discriminator
    def train(genenerator, discriminator, gan_model, dataset, latent_dim, output_path,
            real_samples_multiplier=1.0, fake_samples_multiplier=0.0, discriminator_batches=1,
            n_epochs=100, n_batch=8):
        batch_per_epoch = dataset.shape[0] // n_batch
        half_batch = n_batch // 2
        seed = MuseGAN.generate_latent_points(latent_dim, 87)
        n_steps = batch_per_epoch * n_epochs
        
        history = {'discriminator_real_loss': [],
                'discriminator_fake_loss': [],
                'generator_loss': []}
        for step in range(n_steps):
            epoch = step // batch_per_epoch
            disc_loss_real = 0.0
            disc_loss_fake = 0.0
            for disc_batch in range(discriminator_batches):
                X_real, y_real = MuseGAN.generate_real_samples(dataset, half_batch,
                                                    multiplier=real_samples_multiplier)
                disc_loss_real += discriminator.train_on_batch(X_real, y_real)
                X_fake, y_fake = MuseGAN.generate_fake_samples(genenerator, latent_dim, half_batch,
                                                    multiplier=fake_samples_multiplier)
                disc_loss_fake += discriminator.train_on_batch(X_fake, y_fake)
            disc_loss_real /= discriminator_batches
            disc_loss_fake /= discriminator_batches
            
            X_gan = MuseGAN.generate_latent_points(latent_dim, n_batch)
            y_gan = np.ones((n_batch, 1)) * real_samples_multiplier
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            
            history['discriminator_real_loss'].append(disc_loss_real)
            history['discriminator_fake_loss'].append(disc_loss_fake)
            history['generator_loss'].append(g_loss)
            epoch = step // batch_per_epoch+1
            print('epoch: %d, discriminator_real_loss=%.3f, discriminator_fake_loss=%.3f, generator_loss=%.3f' % (epoch, disc_loss_real, disc_loss_fake, g_loss))

        return history

def create_gan():
    outfile = open('../data/Piano-midi.de.pickle','rb')
    data = pickle.load(outfile)
    print(data["train"][4][19])
    outfile.close()
    dataset=np.zeros((len(data["train"]), 5000, 120))

    for i in range(len(data["train"])): #sample
        for j in range(len(data["train"][i])): #second
            for k in range(len(data["train"][i][j])):
                dataset[i, j, data["train"][i][j][k]]=1; #note
    model = MuseGAN()
    dis = model.define_discriminator((5000, 120))
    gen = model.define_generator(50, (5000, 120))
    gan = model.define_gan(gen, dis)
    hist = model.train(gen, dis, gan, dataset, 50, "xyz")
    return model, dis, gen, gan, hist
