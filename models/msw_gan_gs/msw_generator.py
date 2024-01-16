import tensorflow as tf
from models.progan_normalization import pix_norm

###########################################################
#                                                         #
#         2D Multi-Scale Wasserstein GENERATOR            #
#        with 1-Lipschitz non-monotonic activations       #
#              and Spectral Normalisation                 #
#                                                         #
#          This model is not SOTA, and is here for        #
#                   comparison purposes                   #
#                                                         #
###########################################################

class GeneratorInitial(tf.keras.layers.Layer):
    # First layer of the Multi-Scale Generator
    # It is a simple convolution, but without skip connection, 256 channels and a large kernel size
    def __init__(self, features=256, kernel_size=(9, 9), padding="valid"):
        super(GeneratorInitial, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=features, kernel_size=kernel_size,
                                           strides=(1, 1), activation="relu", padding=padding, name="gen-init-conv")

    def call(self, inputs):
        y = self.conv(inputs)
        y = pix_norm(y)

        return y


class GeneratorBlock(tf.keras.layers.Layer):
    # Intermediate layers of the Multi-Scale Generator
    # This block has: One upsampling layer, two convolutions,
    # one skip connections and a Pixel-Wise Normalisation
    def __init__(self, features, output_features, upsampling_size=(2, 2), kernel_size=(3, 3), padding="valid",
                 name="gen-block"):
        super(GeneratorBlock, self).__init__()
        self.upsample = tf.keras.layers.UpSampling2D(size=upsampling_size,
                                                     interpolation='nearest',
                                                     name=name + "-upsampling")

        self.conv_1 = tf.keras.layers.Conv2D(filters=features, kernel_size=kernel_size,
                                             strides=(1, 1), padding=padding,
                                             activation=tf.keras.layers.LeakyReLU(alpha=0.2),
                                             name=name + "-conv1")

        self.conv_2 = tf.keras.layers.Conv2D(filters=features, kernel_size=kernel_size,
                                             strides=(1, 1), padding=padding,
                                             activation=tf.keras.layers.LeakyReLU(alpha=0.2),
                                             name=name + "-conv2")

        self.conv_skip = tf.keras.layers.Conv2D(filters=output_features, kernel_size=(1, 1),
                                                strides=(1, 1), padding=padding,
                                                activation="softmax", name=name + "-skip-conv")

    def call(self, inputs):
        y = self.upsample(inputs)
        y = self.conv_1(y)
        y = pix_norm(y)
        y = self.conv_2(y)
        y = pix_norm(y)

        y_prime = self.conv_skip(y)

        return y, y_prime


class LastGeneratorBlock(tf.keras.layers.Layer):
    def __init__(self, features, output_features, kernel_size=(3, 3), upsampling=(2, 2), padding="valid",
                 name="gen-fin-block"):
        super(LastGeneratorBlock, self).__init__()
        self.upsample = tf.keras.layers.UpSampling2D(size=upsampling,
                                                     interpolation='bilinear',
                                                     name=name + "-upsampling")

        self.conv_1 = tf.keras.layers.Conv2D(filters=features, kernel_size=kernel_size,
                                             strides=(1, 1), padding=padding,
                                             activation=tf.keras.layers.LeakyReLU(alpha=0.2),
                                             name=name + "-conv1")

        self.conv_2 = tf.keras.layers.Conv2D(filters=output_features, kernel_size=kernel_size,
                                             strides=(1, 1), padding=padding, activation="softmax",
                                             name=name + "-final-conv")

    def call(self, inputs):
        y = self.upsample(inputs)
        y = self.conv_1(y)
        y = pix_norm(y)
        y = self.conv_2(y)

        return y


class MSNWGSGenerator(tf.keras.Model):
    def __init__(self, output_dims, kernel_size=(3, 3), layers_features=None):
        """
        Generator class for Multi-Scale model
        Args:
            output_dims: tuple (h, w, c) giving the dimensions
            kernel_size: dimension of convolution kernels
        """
        super(MSNWGSGenerator, self).__init__()
        if layers_features is None:
            layers_features = [4, 64, 32, 16, 4]
        padding = "same"
        output_features = output_dims[-1]

        self.block_1 = GeneratorInitial(layers_features[0], kernel_size=kernel_size, padding=padding)
        self.block_2 = GeneratorBlock(layers_features[1], output_features, upsampling_size=(1, 1),
                                      kernel_size=kernel_size, padding=padding, name="gen-block-1")
        self.block_3 = GeneratorBlock(layers_features[2], output_features, upsampling_size=(2, 2),
                                      kernel_size=kernel_size, padding=padding, name="gen-block-2")
        self.block_4 = GeneratorBlock(layers_features[3], output_features, upsampling_size=(2, 2),
                                      kernel_size=kernel_size, padding=padding, name="gen-block-3")
        self.block_5 = LastGeneratorBlock(layers_features[-1], output_features, kernel_size=kernel_size,
                                          padding=padding)

        self.generator_blocks = [self.block_1, self.block_2, self.block_3, self.block_4, self.block_5]

    def call(self, inputs):
        outputs = []
        y = self.block_1(inputs)

        for i in range(1, len(self.generator_blocks) - 1):
            y, y_prime = self.generator_blocks[i](y)
            outputs.append(y_prime)

        y = self.generator_blocks[-1](y)
        outputs.append(y)

        return outputs
