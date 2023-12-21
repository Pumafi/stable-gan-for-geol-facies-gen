import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def upsample_block(y, filters, activation, kernel_size=(3, 3), strides=(1, 1), up_size=(2, 2),
                   use_bn=True, name="gen-upsample-block"):
    """
     Function to return a sequence of operations corresponding to a generator block of vanilla GAN;
     a transposed convolution, a batchnorm, and an activation.
     Parameters:
         name:
         y: input slice
         filters: how many channels the output feature representation should have
         activation: the activation function ("sigmoid", "softmax"...)
         kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
         strides: the stride of the convolution
         up_size: the upscaling scale of the layer
         use_bn: bool, use BatchNormalization or not
     Returns:
         The output slice
     """
    y = layers.UpSampling2D(up_size, interpolation='nearest', name=name + "up")(y)
    y = layers.Conv2D(
        filters, kernel_size, strides=strides, padding="same", name=name + "conv")(y)

    if use_bn:
        y = layers.BatchNormalization(momentum=0.9, name=name + "bn")(y)

    y = activation(y)

    return y


def get_generator_model(output_channels, kernel_size=(3, 3), layers_features=None,
                        final_activation=tf.keras.activations.softmax):
    """
    Function to return a Generator Model of DCGAN
    Parameters:
        output_channels: number of channels of the output
        kernel_size: kernel size for convolutions
        layers_features: vector with number of output features at each layer (or number of filters for conv. at layer)
        final_activation: the activation on output layer
    Returns:
        The Generator Model
    """
    if layers_features is None:
        layers_features = [32, 64, 128]

    input = layers.Input(shape=(None, None, 1), name="gen-input")
    y = upsample_block(input, layers_features[0], tf.keras.layers.LeakyReLU(alpha=0.2), kernel_size=kernel_size,
                       up_size=(1, 1),
                       use_bn=True, name="2d-upsample-block-init-")
    for i, nb_features in enumerate(layers_features[1:]):
        y = upsample_block(y, nb_features, tf.keras.layers.LeakyReLU(alpha=0.2), kernel_size=kernel_size,
                           up_size=(2, 2), use_bn=True, name="2d-upsample-block-{0}-".format(i))

    # Final Bloc -> img
    output = upsample_block(y, output_channels, final_activation, kernel_size=(3, 3), up_size=(1, 1), use_bn=True,
                            name="2d-upsample-block-output-")

    g_model = keras.models.Model(input, output, name="generator")
    return g_model
