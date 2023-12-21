import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import colors


def see_pixelwise_error(im1, im2, slice_h, slice_w, mask=None):
    """
    Visualise the pixelwise difference between two slices. If a mask is given, will only compare given pixels.
    Args:
        im1: first image to compare
        im2: second image to compare
        mask: optional mask
        slice_h: height of the images
        slice_w: width of the images
    """

    img_interpolated1 = np.argmax(im1, axis=-1).reshape((slice_h, slice_w))
    img_interpolated2 = np.argmax(im2, axis=-1).reshape((slice_h, slice_w))

    if len(mask.shape) > 2:
        mask = mask.squeeze()

    successes = np.multiply(img_interpolated1, mask) == np.multiply(img_interpolated2, mask)
    successes = np.multiply(successes, mask)

    xs, ys = successes.nonzero()

    plt.scatter(ys, xs, s=5000, linewidth=10, facecolors='none', edgecolors='g')

    errors = np.multiply(img_interpolated1, mask) != np.multiply(img_interpolated2, mask)
    errors = np.multiply(errors, mask)

    xe, ye = errors.nonzero()

    plt.scatter(ye, xe, s=5000, linewidth=10, facecolors='none', edgecolors='r')


def print_conditioned_results(original_pixels, generated_images, mask, nb_simulations, cmap, norm,
                              slice_size=(64, 128)):
    """
    Print results and the pixel wise error together.
    Args:
        original_pixels:
        generated_images:
        mask:
        nb_simulations:
        cmap:
        norm:
        slice_size:
    Returns:
    """
    plt.figure(figsize=(120, 40))
    for i in range(0, nb_simulations):
        # generated Image plotting
        plt.subplot(1, nb_simulations, i + 1)
        plt.gca().set_title('Conditionned result {}'.format(i))
        original_visual = np.argmax(generated_images[i], axis=-1).reshape(slice_size)
        plt.imshow(original_visual, interpolation='nearest', cmap=cmap, norm=norm)
        plt.axis('off')

        fail_patch = mpatches.Patch(color='firebrick', label='Incorrect value')
        success_patch = mpatches.Patch(color='green', label='Correct value')
        plt.legend(handles=[fail_patch, success_patch], bbox_to_anchor=(1.15, 1), prop={'size': 35})

        see_pixelwise_error(original_pixels, generated_images[i],
                            slice_h=slice_size[0], slice_w=slice_size[1],
                            mask=mask)


def compute_probability_map(data):
    """
    Compute a probability map for each facies
    Args:
        data: the data with wich to compute the probability map
    Returns: the proba map and an image with the most probable facies
    """
    proba_map = tf.math.reduce_mean(data, axis=0)
    most_probable_facies = np.argmax(proba_map, axis=-1)

    return proba_map, most_probable_facies


def print_proba_map(proba_map, cmap, norm, samples=None, figsize=(40, 40), plot_variance=False):
    """
    Print the proba map returned by the function compute_probability_map.
    Can also be used to plot the variance instead.
    If conditionning pixels are given, it will print them over the proba or variance map.
    Args:
        proba_map: the proba map returned by the function compute_probability_map
        cmap: color map (matplolib)
        norm: norm for color map (matplotlib)
        samples: the conditioning pixels
        figsize: size of the matplot lib figure
        plot_variance:  plot the variance instead of the proba map
    Returns: None
    """
    fig = plt.figure(figsize=figsize)

    ncomponents = proba_map.shape[-1]
    cmap_mask = colors.ListedColormap(['nipy_spectral'])

    for i in range(ncomponents):
        plt.subplot(1, ncomponents + 1, i + 1)
        plt.axis('off')

        if not plot_variance:
            plt.title("Facies {} probability".format(i))
            img = plt.imshow(proba_map[:, :, i], cmap='nipy_spectral')
            plt.colorbar(fraction=0.025, pad=0.04)
            plt.clim(0, 1)


        else:
            plt.title("Facies {} variability".format(i))
            plt.imshow(proba_map[:, :, i] * (1 - proba_map[:, :, i]), cmap='nipy_spectral')
            plt.colorbar(fraction=0.025, pad=0.04)
            plt.clim(0, 1)

        #
        if samples is not None:
            x_samples, y_samples = (samples[:, :, :, i].squeeze()).nonzero()
            plt.scatter(y_samples, x_samples, linewidth=4, facecolors='none', edgecolors='w')
            plt.scatter(y_samples, x_samples, linewidth=2, facecolors='none', edgecolors='k')

    plt.show()


def show_multi_scale_images_and_probas(array_img, cmap, norm):
    # IMAGES
    plt.figure(figsize=(120, 20))
    plt.axis('off')
    plt.subplot(1, 4, 1)
    slice_h = array_img[-1].numpy().shape[1]
    slice_w = array_img[-1].numpy().shape[2]

    size_high_res = (slice_h, slice_w)
    size_mid_res_1 = (int(slice_h / 2), int(slice_w / 2))
    size_mid_res_2 = (int(slice_h / 4), int(slice_w / 4))
    size_low_res = (int(slice_h / 8), int(slice_w / 8))

    plt.imshow(np.argmax(array_img[-1].numpy(), axis=-1).reshape(size_high_res),
               interpolation='nearest', cmap=cmap, norm=norm)
    plt.subplot(1, 4, 2)
    plt.imshow(np.argmax(array_img[2].numpy(), axis=-1).reshape(size_mid_res_1),
               interpolation='nearest', cmap=cmap, norm=norm)
    plt.subplot(1, 4, 3)
    plt.imshow(np.argmax(array_img[1].numpy(), axis=-1).reshape(size_mid_res_2),
               interpolation='nearest', cmap=cmap, norm=norm)
    plt.subplot(1, 4, 4)
    plt.imshow(np.argmax(array_img[0].numpy(), axis=-1).reshape(size_low_res),
               interpolation='nearest', cmap=cmap, norm=norm)
    plt.show()

    # PROBAS
    for img in array_img:
        proba_map, most_probable_facies = compute_probability_map(img)
        print(type(most_probable_facies))
        print_proba_map(proba_map, cmap, norm)


def conditional_proba_histogram(simulations, mask, conditioning_pixels):
    """
    Given some conditionned simulations, with the corresponding mask and conditionin_pixels, show an histogram of the
    probability of all facies at different locations.
    Red histogram is real data at probability 1 at the true facies value, 0 elsewhere. Blue histogram is the simulated
    probas.
    Args:
        simulations:
        mask:
        conditioning_pixels:

    Returns:

    """
    generated_proba_maps, _ = compute_probability_map(simulations)
    pred_proba_condi = np.squeeze(np.multiply(generated_proba_maps, mask))
    pred_x_idx, pred_y_idx, _ = np.nonzero(pred_proba_condi)
    pred_x_idx, pred_y_idx = list(map(list, zip(*list(set(zip(pred_x_idx, pred_y_idx))))))
    pred_histo_matrix = pred_proba_condi[pred_x_idx, pred_y_idx]
    real_proba_condi = np.squeeze(np.multiply(conditioning_pixels, mask))
    real_x_idx, real_y_idx, _ = np.nonzero(real_proba_condi)
    real_x_idx, real_y_idx = list(map(list, zip(*list(set(zip(real_x_idx, real_y_idx))))))
    real_histo_matrix = real_proba_condi[real_x_idx, real_y_idx]
    plt.figure(figsize=(120, 20))
    for i in range(pred_histo_matrix.shape[0]):
        plt.subplot(1, pred_histo_matrix.shape[0], i + 1)
        plt.bar(["1", "2", "3", "4"], real_histo_matrix[i], alpha=0.5, label='real', color='r')
        plt.bar(["1", "2", "3", "4"], pred_histo_matrix[i], alpha=0.5, label='pred')
    plt.legend(loc='upper right')
    plt.show()


def plot_shannon_entropy_matrix(simulations, conditioning_pixels=None):
    """
    Plot the estimated shannon entropy of a set of simulations
    Args:
        simulations:
        conditioning_pixels:

    Returns:

    """
    generated_proba_maps, _ = compute_probability_map(simulations)
    ncomponents = generated_proba_maps.shape[-1]
    log_matrix = np.log2(generated_proba_maps + 1e-32)
    shannon_entropy_matrix = -np.round_(np.sum(np.multiply(generated_proba_maps, log_matrix), axis=-1), decimals=3)
    plt.title("Shannon Entropy of Facies distribution at each pixel")
    img = plt.imshow(shannon_entropy_matrix, cmap='nipy_spectral')
    plt.colorbar(fraction=0.025, pad=0.04)
    plt.clim(0, np.log2(ncomponents))

    if conditioning_pixels is not None:
        for i in range(ncomponents):
            x_samples, y_samples = (conditioning_pixels[:, :, :, i].squeeze()).nonzero()
            plt.scatter(y_samples, x_samples, linewidth=2, facecolors='none', edgecolors='w')
    plt.show()


def real_facies_proba(simulations, ground_truth_simulation, conditioning_pixels):
    """
    Show a map at every point (even non-conditioning ones) of the probability to get the real facies given some
    simulations and the ground truth.
    Args:
        simulations:
        ground_truth_simulation:
        conditioning_pixels:

    Returns:

    """
    generated_proba_maps, _ = compute_probability_map(simulations)
    real_facies_proba_matrix = np.round(
        np.amax(np.multiply(generated_proba_maps, np.squeeze(ground_truth_simulation)), axis=-1), decimals=4)
    plt.title("Probability of the original image facies at each pixel")
    img = plt.imshow(real_facies_proba_matrix, cmap='nipy_spectral')
    plt.colorbar(fraction=0.025, pad=0.04)
    plt.clim(0, 1)
    ncomponents = generated_proba_maps.shape[-1]

    for i in range(ncomponents):
        x_samples, y_samples = (conditioning_pixels[:, :, :, i].squeeze()).nonzero()
        plt.scatter(y_samples, x_samples, linewidth=2, facecolors='none', edgecolors='w')
    plt.show()
