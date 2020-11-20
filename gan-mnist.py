import matplotlib.pyplot as plt
import numpy as np

from keras.datasets import mnist
from keras.layers import Dense, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam



"""
 Define constants
"""
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
z_dim = 100 # size of the noise vector for generator input


def get_generator(img_shape, z_dim):
    """
        Create a generator
            input: z
            output: 28 *28 * 1 image 
            
    """
    model = Sequential()
    model.add(Dense(128, input_dim=z_dim)) # 128 units
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(img_rows*img_cols*channels, activation='tanh')) # tahnh for crispier images
    model.add(Reshape(img_shape))
    return model

def get_discriminator(img_shape):
    """
        Create a discriminator
            input: 28 * 28 *1 image
            oput: probability whether image is real rather than fake
    """
    model = Sequential()
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(1, activation='sigmoid'))
    return model

def get_gan(generator, discriminator):
    """
        Build Gan
    """
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model


"""compile models"""
disc = get_discriminator(img_shape)
disc.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
disc.trainable = False

gen = get_generator(img_shape, z_dim)
gan = get_gan(gen, disc)
gan.compile(loss='binary_crossentropy', optimizer=Adam())






    


