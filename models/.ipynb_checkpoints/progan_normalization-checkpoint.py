# Normalization functions of Progan and MSGAN (https://arxiv.org/abs/1710.10196)
import tensorflow as tf


class MinibatchStdev(tf.keras.layers.Layer):
    """
    Minibatch Stdev is a normalization layer for 1-Lipschitz (Wasserstein) Discriminator/Critic.
    It was developed for Progressive Growing of GANs (ProGAN).
    We compute the Standard Deviation for each pixels in the input batch, and concatenate the resulting stdev matrix
    to the output of the layer.
    The reasoning is that the discriminator will judge the images not only on realism but also on variaty.
    """
    def __init__(self, **kwargs):
        super(MinibatchStdev, self).__init__(**kwargs)

    # calculate the mean standard deviation across each pixel coord
    def call(self, inputs):
        mean = tf.keras.backend.mean(inputs, axis=0, keepdims=True)
        mean_sq_diff = tf.keras.backend.mean(tf.keras.backend.square(inputs - mean), axis=0, keepdims=True) + 1e-8
        mean_pix = tf.keras.backend.mean(tf.keras.backend.sqrt(mean_sq_diff), keepdims=True)
        shape = tf.keras.backend.shape(inputs)
        output = tf.keras.backend.tile(mean_pix, [shape[0], shape[1], shape[2], 1])
        return tf.keras.backend.concatenate([inputs, output], axis=-1)


def pix_norm(x, epsilon=1e-8):
    """
    Pixel-Normalization
    A version of Batch Normalization but pixel-wise.
    It was developed for Progressive Growing of GANs (ProGAN).
    Args:
        x: output of previous layer
        epsilon: numerical stability term, avoid division by 0

    Returns: scaled x

    """
    scale = tf.sqrt(tf.reduce_mean(x ** 2, axis=-1, keepdims=True) + epsilon)
    return x / scale
