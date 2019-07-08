# Variational Autoencoder

Just a quick example of two variational autoencoders.

The first one, `linear_vae`, is based on a 784 array representation of the MNIST dataset. The second method, `conv_vae` is based on the 2D representation of the MNIST dataset. Each method has just a few layers that make up an encoder and decoder architecture. 

The encoder takes the input and outputs two values, $ \mu $ and $ \sigma $. Then, $ \mu $ and $ \sigma $ are sampled from using the [reparameterization trick](https://arxiv.org/pdf/1312.6114.pdf).

\( z = \mu + \sigma * \epsilon \text{ where } \epsilon \in N(0,1) \)

The decoder is then trained to create the original sample from the sample z. 





