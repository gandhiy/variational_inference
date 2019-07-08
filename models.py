import keras
import tensorflow as tf
import keras.layers as layers


from keras import backend as K 
from keras.models import Model


def conv_vae_loss(inputs, outputs, mu, sig):
    """ 
     convolutional vae loss needs a flatten method
    """
    reconstruction_loss = keras.losses.mse(K.flatten(inputs), K.flatten(outputs))
    reconstruction_loss *= 28
    kl_loss = -0.5*(K.mean(1 + sig - K.square(mu) - K.exp(sig), axis=-1))
    return reconstruction_loss + kl_loss


def vae_loss(inputs, outputs, mu, sig):
    """
     linear vae loss function
    """
    reconstruction_loss = keras.losses.mse(inputs, outputs)
    kl_loss = -0.5*(K.mean(1 + sig - K.square(mu) - K.exp(sig), axis=-1))
    
    return reconstruction_loss + kl_loss


def conv_vae(input_shape, sampling, intermediate_dim, latent_dim, verbose=False):
    """
     input_shape: (-1, image_size, image_size, input_channels)
     sampling: sampling function
     intermediate_dim: Intermediate dimension of the model. 
     latent_dim: number of dimensions for mu and sigma
     verbose: if the model summary should be printed

    """
    # encoder model
    inputs = layers.Input(shape=input_shape, name='enc_in')
    x = layers.Conv2D(16, (3,3), padding='same', strides=2, activation='relu')(inputs)
    x = layers.Conv2D(64, (3,3), padding='same', strides=2, activation='relu')(x)
    # save layer output shape for decoder
    shape = K.int_shape(x)
    x = layers.Flatten()(x)
    # number of parameters after flattening
    flattened_parameters =  K.int_shape(x)[1]
    x = layers.Dense(intermediate_dim, activation='relu')(x)
    mu = layers.Dense(latent_dim, name='mu')(x)
    sig = layers.Dense(latent_dim, name='sigma')(x)
    # sampling with learned mu and sigma
    z = layers.Lambda(sampling, output_shape=(latent_dim, ), name='z')([mu, sig])
    encoder = Model(inputs, [mu, sig, z], name='enc')
    if(verbose):
        encoder.summary()
    
    # decoder model
    dec_inputs = layers.Input(shape=(latent_dim, ), name='dec_in')
    x = layers.Dense(intermediate_dim, activation='relu')(dec_inputs)
    x = layers.Dense(flattened_parameters, activation='relu')(x)
    x = layers.Reshape((shape[1], shape[2], shape[3]))(x)
    x = layers.Conv2DTranspose(16, (3,3), padding='same', strides=2, activation='relu')(x)
    outputs = layers.Conv2DTranspose(1, (3,3), padding='same',  strides=2, activation='sigmoid')(x)
    decoder = Model(dec_inputs, outputs, name='decoder')
    if(verbose):
        decoder.summary()

    outputs = decoder(encoder(inputs)[2])

    # VAE model
    vae = Model(inputs, outputs, name='vae')
    
    if(verbose):
        vae.summary()

    vae.add_loss(conv_vae_loss(inputs, outputs, mu, sig))
    return (vae, encoder, decoder) 




def linear_vae(input_shape, sampling, intermediate_dim, latent_dim = 2, verbose=False):
    """
     Assumes linear input where
     input_shape: (N,)

     sampling: sampling function
     intermediate_dim: Intermediate dimension of the model. 
     latent_dim: number of dimensions for mu and sigma

    """
    if(intermediate_dim < 2): 
        raise ValueError("Need an intermediate dimension greater than 8")
    
    inputs = layers.Input(shape=input_shape, name='enc_in')
    x = layers.Dense(intermediate_dim, activation='relu')(inputs)
    x = layers.Dense(intermediate_dim//2, activation='relu')(x)
    x = layers.Dense(intermediate_dim//4, activation='relu')(x)
    mu = layers.Dense(latent_dim, name='mu')(x)
    sig = layers.Dense(latent_dim, name='sig')(x)

    # reparameterization trick
    z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([mu, sig])
    encoder = Model(inputs, [mu, sig, z], name='enc')
    if(verbose):
        encoder.summary()

    dec_inputs = layers.Input(shape=(latent_dim, ), name='dec_in')
    x = layers.Dense(intermediate_dim//4, activation='relu')(dec_inputs)
    x = layers.Dense(intermediate_dim//2, activation='relu')(x)
    x = layers.Dense(intermediate_dim, activation='relu')(x)
    outputs = layers.Dense(input_shape[0], activation='sigmoid')(x)
    decoder = Model(dec_inputs, outputs, name='decoder')
    if(verbose):
        decoder.summary()

    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')
    vae.add_loss(vae_loss(inputs, outputs, mu, sig))
    return (vae, encoder, decoder) 