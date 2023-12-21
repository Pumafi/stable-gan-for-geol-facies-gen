from deel.lip.activations import GroupSort
from deel.lip.layers import (
    InvertibleDownSampling,
)

from models.spectral_normalization_layers import ConvSN2D
import tensorflow as tf
from tensorflow import keras


def maxsort(x):
    return GroupSort(2)(x)


class StationaryWasserteinApprox(tf.keras.layers.Layer):
    def __init__(self):
        super(StationaryWasserteinApprox, self).__init__()
        self.conv = ConvSN2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='valid')

    def call(self, x):
        y = self.conv(x)
        y = tf.math.reduce_mean(y)
        return y


class InitialDiscriminatorBlock(tf.keras.layers.Layer):
    def __init__(self, features, kernel_size=(3, 3), avg_pooling=(2, 2), padding="same", name="initial-disc-block"):
        super(InitialDiscriminatorBlock, self).__init__()

        self.conv_1 = ConvSN2D(filters=features, kernel_size=kernel_size,
                               strides=(1, 1), padding=padding,
                               activation=maxsort,
                               name=name + "-conv1")

        self.conv_2 = ConvSN2D(filters=features, kernel_size=kernel_size,
                               strides=(1, 1), padding=padding,
                               activation=maxsort,
                               name=name + "-conv2")

        self.avg_pool = InvertibleDownSampling(pool_size=avg_pooling, name=name + "-avg-pool")

    def call(self, inputs):
        y = self.conv_1(inputs)
        y = self.conv_2(y)
        y = self.avg_pool(y)

        return y


class DiscriminatorBlock(tf.keras.layers.Layer):
    def __init__(self, features, kernel_size=(3, 3), avg_pooling=(2, 2), padding="same", name="inter-disc-block"):
        super(DiscriminatorBlock, self).__init__()

        self.conv_1 = ConvSN2D(filters=features, kernel_size=kernel_size,
                               strides=(1, 1), padding=padding,
                               activation=maxsort,
                               name=name + "-conv1")

        self.conv_2 = ConvSN2D(filters=features, kernel_size=kernel_size,
                               strides=(1, 1), padding=padding,
                               activation=maxsort,
                               name=name + "-conv2")

        self.avg_pool = InvertibleDownSampling(pool_size=avg_pooling, name=name + "-avg-pool")

    def call(self, inputs):
        y = self.conv_1(inputs)
        y = self.conv_2(y)
        y = self.avg_pool(y)

        return y


class FinalDiscriminatorBlock(tf.keras.layers.Layer):
    def __init__(self, features, kernel_size_1=(3, 3), kernel_size_2=(3, 3), padding="same", name="final-disc-block"):
        super(FinalDiscriminatorBlock, self).__init__()

        self.conv_1 = ConvSN2D(filters=features, kernel_size=kernel_size_1,
                               strides=(1, 1), padding=padding,
                               activation=maxsort,
                               name=name + "-conv1")

        self.conv_2 = ConvSN2D(filters=features, kernel_size=kernel_size_2,
                               strides=(1, 1), padding=padding,
                               activation=maxsort,
                               name=name + "-conv2")

        self.dense = StationaryWasserteinApprox()

    def call(self, inputs):
        y = self.conv_1(inputs)
        y = self.conv_2(y)
        y = self.dense(y)

        return y


def get_wgs_discriminator_model(input_dims, kernel_size=(3, 3), layers_features=None):
    if layers_features is None:
        layers_features = [16, 32, 64, 128, 256]

    padding = "same"

    x_high_res = tf.keras.layers.Input(shape=(input_dims[0], input_dims[1], input_dims[2]))

    y = InitialDiscriminatorBlock(layers_features[0], kernel_size=kernel_size)(x_high_res)
    y = DiscriminatorBlock(layers_features[1], kernel_size=kernel_size, padding=padding, name="disc-block-1")(y)
    y = DiscriminatorBlock(layers_features[2], kernel_size=kernel_size, padding=padding, name="disc-block-2")(y)
    y = DiscriminatorBlock(layers_features[3], kernel_size=kernel_size, padding=padding, avg_pooling=(1, 1),
                           name="disc-block-3")(y)
    y = FinalDiscriminatorBlock(layers_features[4], kernel_size_1=kernel_size, padding=padding)(y)

    d_model = keras.models.Model(x_high_res, y, name="discriminator")
    return d_model
