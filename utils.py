import matplotlib.pyplot as plt 

from keras.layers import Conv2D
from keras import backend as K

# sampling with mu and sigma
def sampling(args):
    mu, sig = args
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    eps = K.random_normal(shape=(batch, dim))
    return mu + sig*eps

class plotter:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    def plot(self, i):
        if(i < len(self.images)):
            plt.title("Label: {}".format(self.labels[i]))
            plt.imshow(255 - self.images[i], cmap='binary')
        else:
            raise ValueError("index too high")
