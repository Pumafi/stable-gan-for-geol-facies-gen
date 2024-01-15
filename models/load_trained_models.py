from models.wgan.w_generator import get_generator_model
from models.wgan.w_discriminator import get_discriminator_model
from models.wgan.wgan2d import WGAN2D
from models.msw_gan_gs.msw_generator import MSNWGSGenerator
from models.wgan_gs.wgs_generator import WassersteinGSGenerator
from tensorflow import keras

from models.ms_gan.multiscale_generator import MultiScaleGenerator
from models.msw_gan_3d.msw_generator_3d import MultiScaleGenerator3D


def wgan_horizontal(checkpoint_file="../trainedweights/wgan2d/cp-wgan2d_horiz.ckpt",
                    slice_size=(64, 128, 4),
                    noise_shape=(8, 16),
                    nb_convolutions=[16, 32, 64, 128]):
    """
    Load a trained Vanilla Wasserstein Generative Adversarial Network 2D
    Args:
        checkpoint_file: placement of the trained weigths checkpoint
    Returns:
        Both the disc. and gen. trained models
    """

    g_model = get_generator_model(output_channels=slice_size[-1], layers_features=nb_convolutions)
    d_model = get_discriminator_model(slice_size)
    wgan2d = WGAN2D(
        discriminator=d_model,
        generator=g_model,
        latent_shape=noise_shape,
        discriminator_extra_steps=3,
    )
    generator_optimizer = keras.optimizers.Adam(learning_rate=5e-4, beta_1=0.5, beta_2=0.99)
    discriminator_optimizer = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.99)
    wgan2d.compile(d_optimizer=discriminator_optimizer, g_optimizer=generator_optimizer)
    wgan2d.build([None, *slice_size])
    wgan2d.load_weights(checkpoint_file)

    return wgan2d


def load_msgen_horizontal(checkpoint_file="../trainedweights/msgen2dh/cp-gen2d_horizontal_good.ckpt",
                          slice_size = (64, 128, 4),
                          noise_shape = (16, 16)):
    """
    Load a trained Multi-Scale Generative Adversarial Network 2D
    Args:
        checkpoint_file: placement of the trained weigths checkpoint

    Returns:
        The trained generator model
    """
    g_model = MultiScaleGenerator(output_dims=slice_size)
    g_model.build([None, *noise_shape, 1])
    g_model.load_weights(checkpoint_file)

    return g_model


def load_msnwgen_2d_gs_horizontal(checkpoint_file="../trainedweights/msnwgen2d_gs/cp-msnwgen_maxsort_horizontal_good.ckpt",
                                 slice_size = (64, 128, 4),
                                 noise_shape = (8, 16)):
    """
    Load a trained Vanilla Wasserstein Generative Adversarial Network 2D (with Spect. Norm. and GroupSort)
    Args:
        checkpoint_file: placement of the trained weigths checkpoint

    Returns:
        The trained generator model
    """
    g_model = MSNWGSGenerator(output_dims=slice_size)
    g_model.build([None, *noise_shape, 1])
    g_model.load_weights(checkpoint_file)

    return g_model


def load_mswgen_sn_3d_horizontal(checkpoint_file="../trainedweights/mswgen3d/cp-gen2d_horizontal_good.ckpt",
                                 bloc_size=(16, 32, 64, 4),
                                 noise_shape=(2, 4, 8)):
    """
    Load a trained Vanilla Wasserstein Generative Adversarial Network 3D (for volumes) with Spect. Norm.
    Args:
        checkpoint_file: placement of the trained weigths checkpoint

    Returns:
        The trained generator model
    """
    g_model = MultiScaleGenerator3D(output_dims=bloc_size)
    g_model.build([None, *noise_shape, 1])
    g_model.load_weights(checkpoint_file)

    return g_model


def load_wgan_gs_horizontal(checkpoint_file="../trainedweights/wgan_gs/cp-snwgen_horizontal_good.ckpt",
                            slice_size = (64, 128, 4),
                            noise_shape = (8, 16)):
    """
    Load a trained Vanilla Wasserstein Generative Adversarial Network 2D (with GroupSort)
    Args:
        checkpoint_file: placement of the trained weigths checkpoint

    Returns:
        The trained generator model
    """
    g_model = WassersteinGSGenerator(output_dims=slice_size)
    g_model.build([None, *noise_shape, 1])
    g_model.load_weights(checkpoint_file)

    return g_model
