import tensorflow as tf
from tensorflow import keras
from models.spectral_normalization_layers_3d import ConvSN3D
from models.progan_normalization import MinibatchStdev
from models.custom_activation_functions import normalized_swish


class InitialDiscriminatorBlock(tf.keras.layers.Layer):
    def __init__(self, features, kernel_size=(3, 3, 3), avg_pooling=(2, 2, 2), padding="same",
                 name="initial-disc-block"):
        super(InitialDiscriminatorBlock, self).__init__()

        # self.noise = tf.keras.layers.GaussianNoise(1.0)

        self.from_categ = ConvSN3D(filters=features, kernel_size=kernel_size,
                                   strides=(1, 1, 1), padding=padding, name=name + "-from-categories")

        self.mstd = MinibatchStdev()

        self.conv_1 = ConvSN3D(filters=features, kernel_size=kernel_size,
                               strides=(1, 1, 1), padding=padding,
                               activation=normalized_swish,
                               name=name + "-conv1")

        self.conv_2 = ConvSN3D(filters=features, kernel_size=kernel_size,
                               strides=(1, 1, 1), padding=padding,
                               activation=normalized_swish,
                               name=name + "-conv2")

        self.avg_pool = tf.keras.layers.AveragePooling3D(pool_size=avg_pooling, padding="valid",
                                                         name=name + "-avg-pool")

        self.flat = tf.keras.layers.Flatten(name=name + "disc-flatten")
        self.dense = tf.keras.layers.Dense(1, name=name + "disc-dense")

    def call(self, inputs):
        # y = self.noise(inputs)
        y = self.from_categ(inputs)
        y = self.mstd(y)
        y = self.conv_1(y)
        y = self.conv_2(y)
        y = self.avg_pool(y)

        y_prime = self.flat(y)
        y_prime = self.dense(y_prime)

        return y, y_prime


class DiscriminatorBlock(tf.keras.layers.Layer):
    def __init__(self, features, kernel_size=(3, 3, 3), avg_pooling=(2, 2, 2), padding="same", name="inter-disc-block"):
        super(DiscriminatorBlock, self).__init__()

        # self.noise = tf.keras.layers.GaussianNoise(1.0)

        self.concat = tf.keras.layers.Concatenate(axis=-1)

        self.mstd = MinibatchStdev()

        self.conv_1 = ConvSN3D(filters=features, kernel_size=kernel_size,
                               strides=(1, 1, 1), padding=padding,
                               activation=normalized_swish,
                               name=name + "-conv1")

        self.conv_2 = ConvSN3D(filters=features, kernel_size=kernel_size,
                               strides=(1, 1, 1), padding=padding,
                               activation=normalized_swish,
                               name=name + "-conv2")

        self.conv_3 = ConvSN3D(filters=features, kernel_size=kernel_size,
                               strides=(1, 1, 1), padding=padding,
                               activation=normalized_swish,
                               name=name + "-conv3")

        self.avg_pool = tf.keras.layers.AveragePooling3D(pool_size=avg_pooling, padding="valid",
                                                         name=name + "-avg-pool")

        self.flat = tf.keras.layers.Flatten(name=name + "disc-flatten")
        self.dense = tf.keras.layers.Dense(1, name=name + "disc-dense")

    def call(self, inputs):
        # Assert type ?
        assert isinstance(inputs, tuple)

        input_x = inputs[0]
        input_rgb = inputs[1]

        # input_x = self.noise(input_x)

        y = self.concat([input_x, input_rgb])
        y = self.mstd(y)
        y = self.conv_1(y)
        y = self.conv_2(y)
        y = self.conv_3(y)
        y = self.avg_pool(y)

        y_prime = self.flat(y)
        y_prime = self.dense(y_prime)

        return y, y_prime


class FinalDiscriminatorBlock(tf.keras.layers.Layer):
    def __init__(self, features, kernel_size_1=(3, 3, 3), kernel_size_2=(4, 4, 4), padding="same",
                 name="final-disc-block"):
        super(FinalDiscriminatorBlock, self).__init__()

        self.mstd = MinibatchStdev()

        self.conv_1 = ConvSN3D(filters=features, kernel_size=kernel_size_1,
                               strides=(1, 1, 1), padding=padding,
                               activation=normalized_swish,
                               name=name + "-conv1")

        self.conv_2 = ConvSN3D(filters=features, kernel_size=kernel_size_1,
                               strides=(1, 1, 1), padding=padding,
                               activation=normalized_swish,
                               name=name + "-conv2")

        self.conv_3 = ConvSN3D(filters=features, kernel_size=kernel_size_2,
                               strides=(1, 1, 1), padding=padding,
                               activation=normalized_swish,
                               name=name + "-conv3")

        self.flat = tf.keras.layers.Flatten(name="disc-flatten-output")
        self.dense = tf.keras.layers.Dense(1, name="disc-dense-output")

    def call(self, inputs):
        y = self.mstd(inputs)
        y = self.conv_1(y)
        y = self.conv_2(y)
        y = self.conv_3(y)
        y = self.flat(y)
        y = self.dense(y)

        return y


def get_discriminator_mswgan_sn_3d(input_dims, kernel_size=(3, 3, 3), layers_features=None):
    if layers_features is None:
        layers_features = [16, 32, 64, 128, 256]

    padding = "same"

    x_high_res = tf.keras.layers.Input(shape=(input_dims[0], input_dims[1], input_dims[2], input_dims[3]))
    x_3 = tf.keras.layers.Input(
        shape=(input_dims[0] // 2, int(input_dims[1] // 2), int(input_dims[2] // 2), input_dims[3]))
    x_2 = tf.keras.layers.Input(
        shape=(input_dims[0] // 4, int(input_dims[1] // 4), int(input_dims[2] // 4), input_dims[3]))
    x_1 = tf.keras.layers.Input(
        shape=(input_dims[0] // 8, int(input_dims[1] // 8), int(input_dims[2] // 8), input_dims[3]))

    y, y_prime0 = InitialDiscriminatorBlock(layers_features[0], kernel_size=kernel_size)(x_high_res)
    block_input = (y, x_3)
    y, y_prime1 = DiscriminatorBlock(layers_features[1], kernel_size=kernel_size, padding=padding, name="disc-block-1")(
        block_input)
    block_input = (y, x_2)
    y, y_prime2 = DiscriminatorBlock(layers_features[2], kernel_size=kernel_size, padding=padding, name="disc-block-2")(
        block_input)
    block_input = (y, x_1)
    y, y_prime3 = DiscriminatorBlock(layers_features[3], kernel_size=kernel_size, padding=padding,
                                     avg_pooling=(1, 1, 1),
                                     name="disc-block-3")(block_input)
    block_input = y
    y = FinalDiscriminatorBlock(layers_features[4], kernel_size_1=kernel_size, padding=padding)(block_input)

    d_model = keras.models.Model([x_1, x_2, x_3, x_high_res], [y_prime0, y_prime1, y_prime2, y_prime3, y],
                                 name="discriminator")
    return d_model
