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
losses = []
accuracies = []
iteration_checkpoints = []

"""
 Define Hyperparameter constants
"""
batch_size = 128
iterations = 2000
sample_interval = 1000



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

def load_data():
    (train_data, _ ), (_,_) = mnist.load_data()
    train_data = train_data /127.5 - 1.0 # rescales from 0->255 to -1 -> 1
    train_data = np.expand_dims(train_data, axis=3)
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    return train_data, (real, fake)

def train():
    train_data, (real, fake) = load_data()
    for i in range(iterations):
        #get random batch of real images
        idx = np.random.randint(0, train_data.shape[0], batch_size)
        imgs = train_data[idx]

        #generate a batch of fake image
        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = gen.predict(z)

        #train disciminator
        d_loss_real = disc.train_on_batch(imgs, real)
        d_loss_fake = disc.train_on_batch(gen_imgs, fake)
        d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)

        # geenerate a batch of fake images
        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = gen.predict(z)

        # train generator
        g_loss = gan.train_on_batch(z, real)
    
        # save losses and accuracies for plotting
        if (i + 1) % sample_interval == 0:
            losses.append((d_loss, g_loss))
            accuracies.append(100.0 * accuracy)
            iteration_checkpoints.append(i + 1)
            print("%d [D loss: %f, acc.: %.2f%% [G Loss: %f]" % (i + 1, d_loss, 100.0 * accuracy * g_loss))
            sample_images()

def sample_images():
    # random noise
    z = np.random.randint(0, 1, (4*4*z_dim))

    #rescale to 0 and 1
    gen_imgs = generator.predict(z)
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(4, 4, figsize=(4,4), sharey=True, sharex = true)
    cnt = 0
    for i in range(4):
        for j in range(4):
            axs[i,j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axis[i,j].axis('off')
            cnt += 1

    
"""compile models"""
disc = get_discriminator(img_shape)
disc.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
disc.trainable = False

gen = get_generator(img_shape, z_dim)
gan = get_gan(gen, disc)
gan.compile(loss='binary_crossentropy', optimizer=Adam())
train()





    


