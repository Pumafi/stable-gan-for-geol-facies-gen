import tensorflow as tf
from tensorflow import keras
from models.progan_normalization import MinibatchStdev


class InitialDiscriminatorBlock(tf.keras.layers.Layer):
    def __init__(self, features, kernel_size=(3, 3), avg_pooling=(2, 2), padding="same",
                 name="initial-disc-block", add_noise=True):
        super(InitialDiscriminatorBlock, self).__init__()

        self.noise_input = add_noise
        if add_noise:
            self.noise = tf.keras.layers.GaussianNoise(1.0)

        self.from_categ = tf.keras.layers.Conv2D(filters=features, kernel_size=kernel_size,
                                                 strides=(1, 1), padding=padding, name=name + "-from-categories")

        self.mstd = MinibatchStdev()

        self.conv_1 = tf.keras.layers.Conv2D(filters=features, kernel_size=kernel_size,
                                             strides=(1, 1), padding=padding,
                                             activation=tf.keras.layers.LeakyReLU(alpha=0.2),
                                             name=name + "-conv1")

        self.conv_2 = tf.keras.layers.Conv2D(filters=features, kernel_size=kernel_size,
                                             strides=(1, 1), padding=padding,
                                             activation=tf.keras.layers.LeakyReLU(alpha=0.2),
                                             name=name + "-conv2")

        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=avg_pooling, padding="valid",
                                                         name=name + "-avg-pool")

    def call(self, inputs):
        if self.noise_input:
            inputs = self.noise(inputs)
        y = self.from_categ(inputs)
        y = self.mstd(y)
        y = self.conv_1(y)
        y = self.conv_2(y)
        y = self.avg_pool(y)

        return y


class DiscriminatorBlock(tf.keras.layers.Layer):
    def __init__(self, features, kernel_size=(3, 3), avg_pooling=(2, 2), padding="same", name="inter-disc-block"):
        super(DiscriminatorBlock, self).__init__()

        self.concat = tf.keras.layers.Concatenate(axis=-1)

        self.mstd = MinibatchStdev()

        self.conv_1 = tf.keras.layers.Conv2D(filters=features, kernel_size=kernel_size,
                                             strides=(1, 1), padding=padding,
                                             activation=tf.keras.layers.LeakyReLU(alpha=0.2),
                                             name=name + "-conv1")

        self.conv_2 = tf.keras.layers.Conv2D(filters=features, kernel_size=kernel_size,
                                             strides=(1, 1), padding=padding,
                                             activation=tf.keras.layers.LeakyReLU(alpha=0.2),
                                             name=name + "-conv2")

        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=avg_pooling, padding="valid",
                                                         name=name + "-avg-pool")

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

        return y


class FinalDiscriminatorBlock(tf.keras.layers.Layer):
    def __init__(self, features, kernel_size_1=(3, 3), kernel_size_2=(4, 4), padding="same", name="final-disc-block"):
        super(FinalDiscriminatorBlock, self).__init__()

        self.mstd = MinibatchStdev()

        self.conv_1 = tf.keras.layers.Conv2D(filters=features, kernel_size=kernel_size_1,
                                             strides=(1, 1), padding=padding,
                                             activation=tf.keras.layers.LeakyReLU(alpha=0.2),
                                             name=name + "-conv1")

        self.conv_2 = tf.keras.layers.Conv2D(filters=features, kernel_size=kernel_size_2,
                                             strides=(1, 1), padding=padding,
                                             activation=tf.keras.layers.LeakyReLU(alpha=0.2),
                                             name=name + "-conv2")

        self.flat = tf.keras.layers.Flatten(name="disc-flatten-output")
        self.dense = tf.keras.layers.Dense(1, name="disc-dense-output", activation="sigmoid")

    def call(self, inputs):
        y = self.mstd(inputs)
        y = self.conv_1(y)
        y = self.conv_2(y)
        y = self.flat(y)
        y = self.dense(y)

        return y


def get_discriminator_model(input_dims, kernel_size=(3, 3), layers_features=None, add_noise=True):
    if layers_features is None:
        layers_features = [16, 32, 64, 128, 256]

    padding = "same"

    x_high_res = tf.keras.layers.Input(shape=(input_dims[0], input_dims[1], input_dims[2]))
    x_3 = tf.keras.layers.Input(shape=(int(input_dims[0] / 2), int(input_dims[1] / 2), input_dims[2]))
    x_2 = tf.keras.layers.Input(shape=(int(input_dims[0] / 4), int(input_dims[1] / 4), input_dims[2]))
    x_1 = tf.keras.layers.Input(shape=(int(input_dims[0] / 8), int(input_dims[1] / 8), input_dims[2]))

    y = InitialDiscriminatorBlock(layers_features[0], kernel_size=kernel_size, add_noise=add_noise)(x_high_res)
    block_input = (y, x_3)
    y = DiscriminatorBlock(layers_features[1], kernel_size=kernel_size, padding=padding, name="disc-block-1")(
        block_input)
    block_input = (y, x_2)
    y = DiscriminatorBlock(layers_features[2], kernel_size=kernel_size, padding=padding, name="disc-block-2")(
        block_input)
    block_input = (y, x_1)
    y = DiscriminatorBlock(layers_features[3], kernel_size=kernel_size, padding=padding, avg_pooling=(1, 1),
                           name="disc-block-3")(block_input)
    block_input = y
    y = FinalDiscriminatorBlock(layers_features[4], kernel_size_1=kernel_size, padding=padding)(block_input)

    d_model = keras.models.Model([x_1, x_2, x_3, x_high_res], y, name="discriminator")
    return d_model
