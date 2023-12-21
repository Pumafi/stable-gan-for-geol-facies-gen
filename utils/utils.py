import tensorflow as tf
import numpy as np
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt


def generate_noise(batch_size, height, width=None, channels=1):
    """
    Wrapper to easily generate noise for our gans
    Args:
        batch_size: size of the training batch size
        height: height of the latent space
        width: width of the latent space
        channels: number of channels in the latent space

    Returns: The sampled random noise (latent space z)

    """
    if width is None:
        width = height
    return tf.random.normal(shape=(batch_size, height, width, channels))


def correct_percentage(generated_images, original, mask, print_res=False):
    """
    This function is used to compute the percentage of correct answers ogf the conditional generations
    Args:
        generated_images: The conditionally generated images
        original: The original image used for the conditionning
        mask: The pixel mask that was used for conditioning
        print_res: Whether to directly print the percentage or not

    Returns: the percentage of correctly predicted pixels at conditioning points

    """
    slice_h = generated_images.shape[1]
    slice_w = generated_images.shape[2]

    img_interpolated1 = np.argmax(generated_images, axis=-1).reshape((-1, slice_h, slice_w))
    img_interpolated2 = np.argmax(original, axis=-1).reshape((-1, slice_h, slice_w))

    mask = tf.keras.backend.cast(tf.squeeze(tf.repeat(mask, img_interpolated1.shape[0], axis=0)), "int64")

    successes = np.multiply(img_interpolated1, mask) == np.multiply(img_interpolated2, mask)
    successes = np.multiply(successes, mask)

    number_of_simus = generated_images.shape[0]

    n, x, y = mask.numpy().nonzero()
    count_n = len(set(n))
    cond_pts = len(n) / count_n

    ns, xs, ys = successes.nonzero()
    percentage_success = len(ns) / (number_of_simus * cond_pts) * 100

    if print_res:
        print("Percentage of successful predictions {:.2f}%:".format(percentage_success))

    return percentage_success


def correct_percentage_3d(generated_images, original, mask, print_res=False):
    """
    This function is used to compute the percentage of correct answers ogf the conditional generations
    Args:
        generated_images: The conditionally generated images
        original: The original image used for the conditionning
        mask: The pixel mask that was used for conditioning
        print_res: Whether to directly print the percentage or not

    Returns: the percentage of correctly predicted pixels at conditioning points

    """
    slice_d = generated_images.shape[1]
    slice_h = generated_images.shape[2]
    slice_w = generated_images.shape[3]

    img_interpolated1 = np.argmax(generated_images, axis=-1).reshape((-1, slice_d, slice_h, slice_w))
    img_interpolated2 = np.argmax(original, axis=-1).reshape((-1, slice_d, slice_h, slice_w))

    mask = tf.keras.backend.cast(tf.squeeze(tf.repeat(mask, img_interpolated1.shape[0], axis=0)), "int64")

    successes = np.multiply(img_interpolated1, mask) == np.multiply(img_interpolated2, mask)
    successes = np.multiply(successes, mask)

    number_of_simus = generated_images.shape[0]

    n, z, x, y = mask.numpy().nonzero()
    count_n = len(set(n))
    cond_pts = len(n) / count_n

    ns, zs, xs, ys = successes.nonzero()
    percentage_success = len(ns) / (number_of_simus * cond_pts) * 100

    if print_res:
        print("Percentage of successful predictions {:.2f}%:".format(percentage_success))

    return percentage_success


def correct_percentage_histo(generated_images, original, mask, print_res=False):
    """
    This function is used to compute the percentage of correct answers ogf the conditional generations
    Args:
        generated_images: The conditionally generated images
        original: The original image used for the conditionning
        mask: The pixel mask that was used for conditioning
        print_res: Whether to directly print the percentage or not

    Returns: the percentage of correctly predicted pixels at conditioning points

    """
    batch_size = generated_images.shape[0]
    slice_h = generated_images.shape[1]
    slice_w = generated_images.shape[2]

    img_interpolated1 = np.argmax(generated_images, axis=-1).reshape((-1, slice_h, slice_w))
    img_interpolated2 = np.argmax(original, axis=-1).reshape((-1, slice_h, slice_w))

    mask = tf.keras.backend.cast(tf.squeeze(tf.repeat(mask, img_interpolated1.shape[0], axis=0)), "int64")

    successes = np.multiply(img_interpolated1, mask) == np.multiply(img_interpolated2, mask)
    successes = np.multiply(successes, mask)

    number_of_simus = generated_images.shape[0]

    n, x, y = mask.numpy().nonzero()
    counts = Counter(n)
    cond_pts = counts[0]

    ns, xs, ys = successes.nonzero()
    counts = Counter(ns)
    nb_realisation_per_succes_pred = np.array(list(counts.values())) / cond_pts * 100
    bins = len(np.unique(nb_realisation_per_succes_pred))

    sns.histplot(data=nb_realisation_per_succes_pred, bins=bins, discrete=True)
    plt.xlabel("Percentage of correct predictions")
    plt.ylabel("Number of simulations")
    plt.show()
