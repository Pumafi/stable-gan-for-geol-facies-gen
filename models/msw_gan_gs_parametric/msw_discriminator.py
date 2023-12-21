from deel.lip.layers import (
    InvertibleDownSampling,
)
from models.spectral_normalization_layers import ConvSN2D, DenseSN
import tensorflow as tf
from tensorflow import keras
from models.progan_normalization import MinibatchStdev
from models.custom_activation_functions import maxsort


class StationaryWassersteinApprox(tf.keras.layers.Layer):
    """
    Replaces the linear layer at the end of the Disc. No activations are used because Wasserstein can take any value
    """

    def __init__(self):
        super(StationaryWassersteinApprox, self).__init__()
        self.conv = ConvSN2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='valid')

    def call(self, x):
        y = self.conv(x)
        y = tf.math.reduce_mean(y)
        return y


class InitialDiscriminatorBlock(tf.keras.layers.Layer):
    def __init__(self, features, input_dims, kernel_size=(3, 3), avg_pooling=(2, 2), padding="same", name="initial-disc-block"):
        super(InitialDiscriminatorBlock, self).__init__()

        self.from_categ = ConvSN2D(filters=features, kernel_size=kernel_size,
                                   strides=(1, 1), padding=padding, name=name + "-from-categories")

        self.mstd = MinibatchStdev()

        self.conv_1 = ConvSN2D(filters=features, kernel_size=kernel_size,
                               strides=(1, 1), padding=padding,
                               activation=maxsort,
                               name=name + "-conv1")

        self.conv_2 = ConvSN2D(filters=features, kernel_size=kernel_size,
                               strides=(1, 1), padding=padding,
                               activation=maxsort,
                               name=name + "-conv2")

        self.avg_pool = InvertibleDownSampling(pool_size=avg_pooling, name=name + "-avg-pool")
        self.last_layer = StationaryWassersteinApprox()

        # Second inputs (parameters) layers: Contrarely to generator we do not use depthwise convolution for 2
        # reasons : 1. the discriminator does not need to be stationnary
        # 2. the discriminator needs to be \
        # 1-Lipschitz, and we're not sure how the depthwise convolution would impact that
        self.fc = DenseSN(input_dims[0] * input_dims[1], activation="swish")
        self.reshape = tf.keras.layers.Reshape((input_dims[0], input_dims[1], 1))
        self.conv_condi = ConvSN2D(filters=features, kernel_size=kernel_size,
                                   strides=(1, 1), activation="swish", padding=padding,
                                   name="conditional-init-conv")
        self.concat = tf.keras.layers.Concatenate(axis=-1)

    def call(self, inputs):
        assert isinstance(inputs, tuple)
        input_img = inputs[0]
        input_params = inputs[1]
        y = self.from_categ(input_img)
        y = self.mstd(y)
        y = self.conv_1(y)

        # parameters (IDEA: why not reuse the embedding of generator ?)
        embedded_parameters = self.fc(input_params)
        embedded_parameters = self.reshape(embedded_parameters)
        # Here we do not generate filters, directly features we concat to previous output features
        embedded_parameters = self.conv_condi(embedded_parameters)

        # Concat parameters
        y = self.concat([y, embedded_parameters])
        y = self.conv_2(y)
        y = self.avg_pool(y)

        y_prime = self.last_layer(y)
        return y, y_prime


class DiscriminatorBlock(tf.keras.layers.Layer):
    def __init__(self, features, kernel_size=(3, 3), avg_pooling=(2, 2), padding="same", name="inter-disc-block"):
        super(DiscriminatorBlock, self).__init__()

        self.concat = tf.keras.layers.Concatenate(axis=-1)

        self.mstd = MinibatchStdev()

        self.conv_1 = ConvSN2D(filters=features, kernel_size=kernel_size,
                               strides=(1, 1), padding=padding,
                               activation=maxsort,
                               name=name + "-conv1")

        self.conv_2 = ConvSN2D(filters=features, kernel_size=kernel_size,
                               strides=(1, 1), padding=padding,
                               activation=maxsort,
                               name=name + "-conv2")

        # Maybe the difference of training steps between GEN and DISC is because of complexity.
        # Increasing or decreasing it might help ?? I'll test more one day :)
        # self.conv_3 = ConvSN2D(filters=features, kernel_size=kernel_size,
        #                                     strides=(1, 1), padding=padding,
        #                                     activation=maxsort,
        #                                     name=name + "-conv3")

        self.avg_pool = InvertibleDownSampling(pool_size=avg_pooling, name=name + "-avg-pool")

        self.last_layer = StationaryWassersteinApprox()

    def call(self, inputs):
        # Assert type ?
        assert isinstance(inputs, tuple)

        input_x = inputs[0]
        input_rgb = inputs[1]

        y = self.concat([input_x, input_rgb])
        y = self.mstd(y)
        y = self.conv_1(y)
        y = self.conv_2(y)
        y = self.avg_pool(y)

        y_prime = self.last_layer(y)
        return y, y_prime


class FinalDiscriminatorBlock(tf.keras.layers.Layer):
    def __init__(self, features, kernel_size_1=(3, 3), kernel_size_2=(4, 4), padding="same", name="final-disc-block"):
        super(FinalDiscriminatorBlock, self).__init__()

        self.mstd = MinibatchStdev()

        self.conv_1 = ConvSN2D(filters=features, kernel_size=kernel_size_1,
                               strides=(1, 1), padding=padding,
                               activation=maxsort,
                               name=name + "-conv1")

        self.conv_2 = ConvSN2D(filters=features, kernel_size=kernel_size_2,
                               strides=(1, 1), padding=padding,
                               activation=maxsort,
                               name=name + "-conv2")

        self.last_layer = StationaryWassersteinApprox()

    def call(self, inputs):
        y = self.mstd(inputs)
        y = self.conv_1(y)
        y = self.conv_2(y)
        y = self.last_layer(y)

        return y


def get_msnwgs_discriminator_model(input_dims, kernel_size=(3, 3), layers_features=None):
    if layers_features is None:
        layers_features = [16, 32, 64, 128, 256]

    padding = "same"

    # Would be cool to put the nb of parameters as a Model calling parameter
    cond_input = tf.keras.layers.Input(shape=(2,))

    x_high_res = tf.keras.layers.Input(shape=(input_dims[0], input_dims[1], input_dims[2]))
    x_3 = tf.keras.layers.Input(shape=(int(input_dims[0] / 2), int(input_dims[1] / 2), input_dims[2]))
    x_2 = tf.keras.layers.Input(shape=(int(input_dims[0] / 4), int(input_dims[1] / 4), input_dims[2]))
    x_1 = tf.keras.layers.Input(shape=(int(input_dims[0] / 8), int(input_dims[1] / 8), input_dims[2]))

    block_input = (x_high_res, cond_input)
    y, y_prime0 = InitialDiscriminatorBlock(layers_features[0], input_dims=input_dims, kernel_size=kernel_size)(block_input)
    block_input = (y, x_3)
    y, y_prime1 = DiscriminatorBlock(layers_features[1], kernel_size=kernel_size, padding=padding, name="disc-block-1")(
        block_input)
    block_input = (y, x_2)
    y, y_prime2 = DiscriminatorBlock(layers_features[2], kernel_size=kernel_size, padding=padding, name="disc-block-2")(
        block_input)
    block_input = (y, x_1)
    y, y_prime3 = DiscriminatorBlock(layers_features[3], kernel_size=kernel_size, padding=padding, avg_pooling=(1, 1),
                                     name="disc-block-3")(block_input)
    block_input = y
    y = FinalDiscriminatorBlock(layers_features[4], kernel_size_1=kernel_size, padding=padding)(block_input)

    d_model = keras.models.Model([cond_input, x_1, x_2, x_3, x_high_res], [y_prime0, y_prime1, y_prime2, y_prime3, y],
                                 name="discriminator")
    return d_model
