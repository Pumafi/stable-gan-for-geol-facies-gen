from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class GANMonitor(keras.callbacks.Callback):
    """
    Callback for GAN training. Saves pictures of the Generator during training to show progress.
    To call with the function model.fit(..., callback=[GANMonitor(latent_shape, cmap, norm, slice_h, slice_w, num_img)])
    """

    def __init__(self, latent_shape, cmap, norm, slice_h, slice_w, num_random_img=2, num_fixed_img=2, fixed_seed=0):
        super().__init__()
        self.num_img = num_random_img
        self.num_fixed_img = num_fixed_img
        self.cmap = cmap
        self.norm = norm
        self.latent_shape = latent_shape
        self.slice_h = slice_h
        self.slice_w = slice_w
        # Define some fixed vectors at initialization to hva constant tests
        if self.num_fixed_img:
            self.fixed_random_latent_vectors = tf.random.normal(shape=(self.num_fixed_img, self.latent_shape[0],
                                                                       self.latent_shape[1], 1), seed=fixed_seed)

    def on_epoch_end(self, epoch, logs=None):
        # Created images with random latent vector
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_shape[0], self.latent_shape[1], 1))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images = np.argmax(generated_images, axis=-1).reshape((-1, self.slice_h, self.slice_w))

        for i in range(self.num_img):
            img = generated_images[i]
            plt.axis('off')
            plt.imshow(img, interpolation='nearest', cmap=self.cmap, norm=self.norm)
            plt.savefig("generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch))
        plt.close()

        # Fixed images with fixed seed to allow better quality testing
        if self.num_fixed_img:
            fixed_generated_images = self.model.generator(self.fixed_random_latent_vectors)
            fixed_generated_images = np.argmax(fixed_generated_images, axis=-1).reshape(
                (-1, self.slice_h, self.slice_w))

            for i in range(self.num_fixed_img):
                img = fixed_generated_images[i]
                plt.axis('off')
                plt.imshow(img, interpolation='nearest', cmap=self.cmap, norm=self.norm)
                plt.savefig("fixed_generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch))
            plt.close()


class MSGANMonitor(keras.callbacks.Callback):
    """
    Callback for GAN training. Saves pictures of the Generator during training to show progress.
    To call with the function model.fit(..., callback=[GANMonitor(latent_shape, cmap, norm, slice_h, slice_w, num_img)])
    """

    def __init__(self, latent_shape, cmap, norm, slice_h, slice_w, num_random_img=2):
        super().__init__()
        self.num_img = num_random_img
        self.cmap = cmap
        self.norm = norm
        self.latent_shape = latent_shape
        self.slice_h = slice_h
        self.slice_w = slice_w
        # Define some fixed vectors at initialization to hva constant tests

    def on_epoch_end(self, epoch, logs=None):
        # Created images with random latent vector
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_shape[0], self.latent_shape[1], 1))
        generated_images = self.model.generator(random_latent_vectors)

        size_high_res = (self.slice_h, self.slice_w)
        size_mid_res_1 = (int(self.slice_h / 2), int(self.slice_w / 2))
        size_mid_res_2 = (int(self.slice_h / 4), int(self.slice_w / 4))
        size_low_res = (int(self.slice_h / 8), int(self.slice_w / 8))

        for i in range(self.num_img):
            plt.axis('off')
            plt.subplot(1, 4, 1)
            plt.imshow(np.argmax(generated_images[-1][i].numpy(), axis=-1).reshape(size_high_res),
                       interpolation='nearest', cmap=self.cmap, norm=self.normorm)
            plt.subplot(1, 4, 2)
            plt.imshow(np.argmax(generated_images[2][i].numpy(), axis=-1).reshape(size_mid_res_1),
                       interpolation='nearest', cmap=self.cmap, norm=self.norm)
            plt.subplot(1, 4, 3)
            plt.imshow(np.argmax(generated_images[1][i].numpy(), axis=-1).reshape(size_mid_res_2),
                       interpolation='nearest', cmap=self.cmap, norm=self.norm)
            plt.subplot(1, 4, 4)
            plt.imshow(np.argmax(generated_images[0][i].numpy(), axis=-1).reshape(size_low_res),
                       interpolation='nearest', cmap=self.cmap, norm=self.norm)

            plt.savefig("fixed_generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch))
        plt.close()


class CheckpointSaving(keras.callbacks.Callback):
    """
    Callback for GAN training. Saves the weights of the generator and discriminator every *saving_frequency* epochs.
    To call with the function model.fit(..., callback=[CheckpointSaving()])
    """

    def __init__(self, generator_save_filepath="./trainedweights/new_generator/cp-new_generator.ckpt",
                 discriminator_save_filepath="./trainedweights/new_discriminator/cp-new_discriminator.ckpt",
                 saving_frequency=10):
        """
        Init the callback
        Args:
            generator_save_filepath: path to the generator saving folder
            discriminator_save_filepath: path to the discriminator saving folder
            saving_frequency: when the model will save
        """
        super().__init__()
        self.generator_save_filepath = generator_save_filepath
        self.discriminator_save_filepath = discriminator_save_filepath
        self.saving_frequency = saving_frequency

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.saving_frequency == 0:
            self.model.generator.save_weights(self.generator_save_filepath)
            self.model.discriminator.save_weights(self.discriminator_save_filepath)
