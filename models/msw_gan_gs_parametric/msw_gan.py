import tensorflow as tf
from tensorflow import keras
from utils.utils import generate_noise
from tensorflow import image

tf.config.run_functions_eagerly(True)

###########################################################
#                                                         #
#            Parametric Multi-Scale GAN                   #
#                                                         #
#          This model is not SOTA, and is here for        #
#                   comparison purposes                   #
#                                                         #
###########################################################

def discriminator_loss(real_img, fake_img):
    # Discriminator loss Wasserstein
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


def generator_loss(fake_img):
    # Generator loss, Wasserstein
    return -tf.reduce_mean(fake_img)


class MSWGANGS(keras.Model):
    """
    MSWGAN2D (Wasserstein Generative Adversarial Network) model with Multi-Scale, Spectral Normalisation and Maxsort
    Conditional Implementation
    """

    def __init__(
            self,
            discriminator,
            generator,
            latent_shape,
            discriminator_extra_steps=3,
            generator_extra_steps=1,
            real_image_resize_method=image.ResizeMethod.BILINEAR
    ):
        super(MSWGANGS, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_shape = latent_shape
        self.d_steps = discriminator_extra_steps
        self.g_steps = generator_extra_steps
        self.real_image_resize_method = real_image_resize_method

    def compile(self, d_optimizer, g_optimizer, d_loss_fn=discriminator_loss, g_loss_fn=generator_loss):
        super(MSWGANGS, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    @tf.function
    def call(self, x):
        return None

    @tf.function
    def train_step(self, real_images_with_params):

        real_images = real_images_with_params[0]
        params = real_images_with_params[1]

        batch_size = tf.shape(real_images)[0]

        x_1 = tf.image.resize(real_images, [self.latent_shape[0], self.latent_shape[1]],
                              method=self.real_image_resize_method)
        x_2 = tf.image.resize(real_images, [self.latent_shape[0] * 2, self.latent_shape[1] * 2],
                              method=self.real_image_resize_method)
        x_3 = tf.image.resize(real_images, [self.latent_shape[0] * 4, self.latent_shape[1] * 4],
                              method=self.real_image_resize_method)
        x_high_res = tf.image.resize(real_images, [self.latent_shape[0] * 8, self.latent_shape[1] * 8],
                                     method=self.real_image_resize_method)

        real_images = [params, x_1, x_2, x_3, x_high_res]

        # Discriminator training
        for i in range(self.d_steps):

            random_latent_vectors = generate_noise(batch_size, self.latent_shape[0], self.latent_shape[1],
                                                   self.latent_shape[-1])
            with tf.GradientTape() as tape:
                gen_inputs = (random_latent_vectors, params)
                fake_images = self.generator(gen_inputs, training=True)
                fake_images = [params, *fake_images]
                fake_logits = self.discriminator(fake_images, training=True)
                real_logits = self.discriminator(real_images, training=True)

                d_loss = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Regularization term to prevent the loss from exploding and causing numerical issues
                d_loss = d_loss + 0.0001 * tf.reduce_mean(tf.square(tf.concat([fake_logits, real_logits], axis=0)))

            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))

        # Generator training
        for i in range(self.g_steps):
            random_latent_vectors = generate_noise(batch_size, self.latent_shape[0], self.latent_shape[1],
                                                   self.latent_shape[-1])
            with tf.GradientTape() as tape:
                gen_inputs = (random_latent_vectors, params)
                generated_images = self.generator(gen_inputs, training=True)
                generated_images = [params, *generated_images]
                gen_img_logits = self.discriminator(generated_images, training=True)
                g_loss = self.g_loss_fn(gen_img_logits)

            gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
            self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}
