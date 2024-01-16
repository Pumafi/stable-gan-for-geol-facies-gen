import tensorflow as tf
from tensorflow import keras
from utils.utils import generate_noise
from tensorflow import image

tf.config.run_functions_eagerly(True)

###########################################################
#                                                         #
#                    Multi-Scale GAN                      #
#                                                         #
#          This model is not SOTA, and is here for        #
#                   comparison purposes                   #
#                                                         #
###########################################################

def discriminator_loss(real_img, fake_img):
    # Discriminator loss, MinMax
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    real_loss = cross_entropy(tf.ones_like(real_img), real_img)
    fake_loss = cross_entropy(tf.zeros_like(fake_img), fake_img)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_img):
    # Discriminator loss, MinMax
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    return cross_entropy(tf.ones_like(fake_img), fake_img)


class MSGAN(keras.Model):
    """
    MSGAN2D (Multiscale Generative Adversarial Network) model with Gaussian Noise
    """

    def __init__(
            self,
            discriminator,
            generator,
            latent_shape,
            discriminator_extra_steps=3,
            generator_extra_steps=1,
            gp_weight=10.0,
            real_image_resize_method=image.ResizeMethod.BILINEAR
    ):
        """
        Init the Multi-scale GAN model
        Args:
            discriminator: an already created multi-scale discriminator model (multiscale_discriminator.py)
            generator: an already created multi-scale generator model (multiscale_generator.py)
            latent_shape: size of the input noise (latent space)
            discriminator_extra_steps: how many times the discriminator is trained by iteration
            generator_extra_steps:  how many times the generator is trained by iteration
            gp_weight: weight of the Gradient Penalty for 1-Lipchitz models (NOT USED)
            real_image_resize_method: reshape method
        """
        super(MSGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_shape = latent_shape
        self.d_steps = discriminator_extra_steps
        self.g_steps = generator_extra_steps
        self.gp_weight = gp_weight
        self.real_image_resize_method = real_image_resize_method

    def compile(self, d_optimizer, g_optimizer, d_loss_fn=discriminator_loss, g_loss_fn=generator_loss):
        super(MSGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    @tf.function
    def call(self, x):
        batch_size = tf.shape(x)[0]
        random_latent_vectors = generate_noise(batch_size, self.latent_shape[0], self.latent_shape[1],
                                               self.latent_shape[-1])
        generated_images = self.generator(random_latent_vectors, training=True)
        return generated_images

    @tf.function
    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        batch_size = tf.shape(real_images)[0]

        # Multiscale images inputs
        x_1 = tf.image.resize(real_images, [self.latent_shape[0], self.latent_shape[1]],
                              method=self.real_image_resize_method)
        x_2 = tf.image.resize(real_images, [self.latent_shape[0] * 2, self.latent_shape[1] * 2],
                              method=self.real_image_resize_method)
        x_3 = tf.image.resize(real_images, [self.latent_shape[0] * 4, self.latent_shape[1] * 4],
                              method=self.real_image_resize_method)
        x_high_res = tf.image.resize(real_images, [self.latent_shape[0] * 8, self.latent_shape[1] * 8],
                                     method=self.real_image_resize_method)

        real_images = [x_1, x_2, x_3, x_high_res]

        for i in range(self.d_steps):
            random_latent_vectors = generate_noise(batch_size, self.latent_shape[0], self.latent_shape[1],
                                                   self.latent_shape[-1])
            with tf.GradientTape() as tape:
                fake_images = self.generator(random_latent_vectors, training=True)
                fake_logits = self.discriminator(fake_images, training=True)
                real_logits = self.discriminator(real_images, training=True)

                d_loss = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)

            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        for i in range(self.g_steps):
            random_latent_vectors = generate_noise(batch_size, self.latent_shape[0], self.latent_shape[1],
                                                   self.latent_shape[-1])
            with tf.GradientTape() as tape:
                generated_images = self.generator(random_latent_vectors, training=True)
                gen_img_logits = self.discriminator(generated_images, training=True)
                g_loss = self.g_loss_fn(gen_img_logits)

            gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
            self.g_optimizer.apply_gradients(
                zip(gen_gradient, self.generator.trainable_variables)
            )

        return {"d_loss": d_loss, "g_loss": g_loss}
