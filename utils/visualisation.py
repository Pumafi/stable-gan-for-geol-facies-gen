import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from utils.utils import generate_noise


def get_color_map(number_of_categories=4):
    """
    Get the matplotlib colormap and norm for images visualisation
    Args:
        number_of_categories: number of facies in the slice

    Returns: cmap, norm

    """
    if number_of_categories == 4:
        cmap = colors.ListedColormap(["#FF8000", "#CBCB33", "#9898E5", "#66CB33"])
        bounds = [-0.1, 0.9, 1.9, 2.9, 3.9]
    elif number_of_categories == 5:
        cmap = colors.ListedColormap(["#000000", "#5387AD", "#7DD57E", "#F1E33E", "#C70000"])
        bounds = [-0.1, 0.9, 1.9, 2.9, 3.9, 4.9]
    else:  # 9
        cmap = colors.ListedColormap(
            ["#000000", "#294255", "#5387AD", "#6DB6B1", "#7DD57E", "#B5DF5D", "#F1E33E", "#F77420", "#C70000"])
        bounds = [-0.1, 0.9, 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9]

    norm = colors.BoundaryNorm(bounds, cmap.N)

    return cmap, norm


def visualise_slice(img, cmap, norm, slice_h, slice_w, figsize=(40, 40)) -> None:
    """
    Visualise a slice
    Args:
        img: the image to visualise, must be of dimensions (slice_h, slice_w, ...)
        cmap: the matplotlib colormap
        norm: the matplotlib norm for the color map
        slice_h: height of the picture
        slice_w: width of the picture
        figsize: size of the matplotlib figure

    Returns: None

    """
    plt.figure(figsize=figsize)
    img_interpolated = np.argmax(img, axis=-1).reshape((slice_h, slice_w))
    plt.axis('off')
    plt.imshow(img_interpolated, interpolation='nearest', cmap=cmap, norm=norm)
    plt.show()


def visualise_slice_components(img, cmap, norm, slice_h, slice_w, figsize=(40, 40)) -> None:
    """
    Visualise a slice and its different components, in a separate manner
    Args:
        img: the image to visualise, must be of dimensions (slice_h, slice_w, ...)
        cmap: the matplotlib colormap
        norm: the matplotlib norm for the color map
        slice_h: height of the picture
        slice_w: width of the picture
        figsize: size of the matplotlib figure

    """
    plt.figure(figsize=figsize)

    ncomponents = img.shape[-1]
    img_interpolated = np.argmax(img, axis=-1).reshape((slice_h, slice_w))

    plt.subplot(1, ncomponents + 2, 1)
    plt.axis('off')
    plt.imshow(img_interpolated, interpolation='nearest', cmap=cmap, norm=norm)
    for i in range(ncomponents):
        plt.subplot(1, ncomponents + 2, i + 2)
        plt.axis('off')
        plt.imshow(img[:, :, i])
    plt.show()


def paint_image(generator, generation_height: int, generation_width: int, lag: int = 5, batch_size: int = 1):
    """
    Visualise multiple image with a partly common noise.
    The images will be generated using the generator passed as an argument, and will generate
    generation_height * generation_width images.
    This is a useful function to visualise side effects.
    Args:
        generator: a trained generator
        generation_height: how many images to create vertically
        generation_width: how many images to create horizontally
        lag: the common part in the noise. Must be found empirically according to how the generator was trained
        batch_size:

    Returns:

    """
    _, noise_height, noise_width, noise_channels = generator.layers[0].output_shape[0]
    _, output_height, output_width, output_channels = generator.layers[-1].output_shape

    latent_table = generate_noise(batch_size, noise_height * generation_height, noise_width * generation_width,
                                  noise_channels)
    result = np.zeros((output_height + (generation_height - 1) * lag,
                       output_width + (generation_width - 1) * lag,
                       output_channels
                       ))

    for i in range(generation_height):
        for j in range(generation_width):
            generated_slice = generator(latent_table[:, i:(noise_height + i), j:(noise_width + j), :])
            result[i * lag:output_height + i * lag, j * lag:output_width + j * lag, :] = generated_slice

    return result


def show_multi_scale_images(array_img, cmap, norm, img_dim=None, i=0, epoch=0, save_img=False):
    plt.figure(figsize=(120, 20))
    plt.axis('off')
    plt.subplot(1, 4, 1)
    slice_h = array_img[-1].numpy().shape[1]
    slice_w = array_img[-1].numpy().shape[2]
    img_dim = (slice_h, slice_w)
    plt.imshow(np.argmax(array_img[-1].numpy(), axis=-1).reshape(img_dim), interpolation='nearest', cmap=cmap,
               norm=norm)
    plt.subplot(1, 4, 2)
    plt.imshow(np.argmax(array_img[2].numpy(), axis=-1).reshape((int(img_dim[0] / 2), int(img_dim[1] / 2))),
               interpolation='nearest', cmap=cmap, norm=norm)
    plt.subplot(1, 4, 3)
    plt.imshow(np.argmax(array_img[1].numpy(), axis=-1).reshape((int(img_dim[0] / 4), int(img_dim[1] / 4))),
               interpolation='nearest', cmap=cmap, norm=norm)
    plt.subplot(1, 4, 4)
    plt.imshow(np.argmax(array_img[0].numpy(), axis=-1).reshape((int(img_dim[0] / 8), int(img_dim[1] / 8))),
               interpolation='nearest', cmap=cmap, norm=norm)

    if save_img:
        plt.savefig("fixed_generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch))
        plt.close()
    else:
        plt.show()


def multi_scale_paint_image(generator, generation_height: int, generation_width: int, noise_shape,
                            lag: int = 5, output_lag=5, batch_size: int = 1, border_effect_size=35):
    """
    Visualise multiple image with a partly common noise.
    The images will be generated using the generator passed as an argument, and will generate
    generation_height * generation_width images.
    This is a useful function to visualise side effects.
    Args:

        noise_shape:
        generator: a trained generator
        generation_height: how many images to create vertically
        generation_width: how many images to create horizontally
        lag: the common part in the noise
        output_lag: the common part in the output
        border_effect_size: size of the bordures effects
        batch_size:

    Returns:

    """
    _, noise_height, noise_width, noise_channels = (1, *noise_shape, 1)
    _, output_height, output_width, output_channels = (1, 64, 128, 4)

    latent_table = generate_noise(batch_size, noise_height * generation_height, noise_width * generation_width,
                                  noise_channels)
    result = np.zeros((64 * generation_height,
                       (output_width - border_effect_size * 2 - output_lag) + (output_lag * generation_width),
                       output_channels
                       ))

    for i in range(generation_height):
        for j in range(generation_width):
            generated_slice = generator(latent_table[:, i:(noise_height + i), j:(noise_width + j), :])[-1]
            result[i * output_lag:output_height + i * output_lag,
            j * output_lag:output_width + j * output_lag - (2 * border_effect_size), :] = generated_slice[:, :,
                                                                                          border_effect_size:-border_effect_size]

    return result
