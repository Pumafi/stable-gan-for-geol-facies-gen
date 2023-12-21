import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def conv_block(y, filters, activation, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bn=False,
               use_dropout=False, drop_value=0.5, name="disc-conv-block-"):
    """
    Function to return a sequence of operations corresponding to a critic block of DCGAN;
    a convolution, a batchnorm, and an activation.
    Parameters:
        name:
        y: input slice
        filters: how many channels the output feature representation should have
        activation: the activation function ("sigmoid", "softmax"...)
        kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
        strides: the stride of the convolution
        padding: "valid" for no padding, "same" to keep same size
        use_bn: bool, use BatchNormalization or not
        use_dropout: bool, use dropout or not
        drop_value: dropout value
    Returns:
        The output slice
    """
    y = layers.Conv2D(
        filters, kernel_size, strides=strides, padding=padding, name=name + "conv"
    )(y)
    if use_bn:
        y = layers.BatchNormalization(name=name + "bn")(y)
    y = activation(y)
    y = layers.AveragePooling2D(pool_size=(2, 2), name=name + "-avg-pool")(y)
    if use_dropout:
        y = layers.Dropout(drop_value, name=name + "dropout")(y)
    return y


def get_discriminator_model(slice_shape, kernel_size=(3, 3), layers_features=None):
    """
    Function to return a Discriminator Model of DCGAN
    Parameters:
        layers_features:
        kernel_size:
        slice_shape: Shape of the input slice (h, w, c)
    Returns:
        The Discriminator Model
    """

    if layers_features is None:
        layers_features = [64, 128, 256, 512]

    # No critic batch normalization (See  I.Gulrajani, 2017, Imrpoved Training of GANs)
    use_batch_norm = False
    activation_function = tf.keras.layers.LeakyReLU(alpha=0.2)

    img_input = layers.Input(shape=slice_shape)
    y = conv_block(img_input, layers_features[0], kernel_size=kernel_size, use_bn=use_batch_norm,
                   activation=activation_function, name="disc-conv-block-1-")
    y = conv_block(y, layers_features[1], kernel_size=kernel_size, use_bn=use_batch_norm,
                   activation=activation_function, name="disc-conv-block-2-")
    y = conv_block(y, layers_features[2], kernel_size=kernel_size, use_bn=use_batch_norm,
                   activation=activation_function, name="disc-conv-block-3-")
    y = conv_block(y,  layers_features[3], kernel_size=kernel_size, use_bn=use_batch_norm,
                   activation=activation_function, name="disc-conv-block-4-")

    y = layers.Flatten(name="disc-flatten-output")(y)
    y = layers.Dropout(0.2, name="disc-dropout-output")(y)
    y = layers.Dense(1, name="disc-dense-output")(y)

    d_model = keras.models.Model(img_input, y, name="discriminator")
    return d_model
