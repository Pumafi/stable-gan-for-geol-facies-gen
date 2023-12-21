from tqdm.auto import tqdm
import tensorflow as tf
import numpy as np

def loss_function_crossentropy(generated_noise, generated_images, conditionning_pixels_vector, mask,
                               index_conditionning_data, weights):
    # l2
    mask = tf.repeat(mask, generated_images.shape[0], axis=0)
    fake_pixels = tf.squeeze(tf.keras.layers.Multiply()([generated_images, mask]))
    # real_pixels = tf.repeat(real_pixels, fake_pixels.shape[0], axis=0)

    fake_pixels_flat = tf.reshape(fake_pixels, (
    fake_pixels.shape[0], fake_pixels.shape[1] * fake_pixels.shape[2] * fake_pixels.shape[3], -1))
    fake_pixels_cond_values = tf.gather(fake_pixels_flat, index_conditionning_data, batch_dims=1)
    real_pixels_cond_values = tf.gather(conditionning_pixels_vector, index_conditionning_data, batch_dims=1)

    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    neg_log_likelihood = cce(real_pixels_cond_values, fake_pixels_cond_values)
    neg_log_likelihood = tf.reduce_sum(neg_log_likelihood, axis=[1])

    prior = 0.5 * tf.reduce_sum(tf.square(generated_noise), axis=[1, 2, 3])

    return tf.math.reduce_mean(weights * tf.math.reduce_mean(neg_log_likelihood, axis=-1)), tf.math.reduce_mean(
        weights * prior)


@tf.function
def tf_euclidean_distance_matrix(vectorised_noise):
    vectorised_noise_a = tf.expand_dims(vectorised_noise, 1)
    vectorised_noise_b = tf.expand_dims(vectorised_noise, 0)
    distances_matrix = tf.math.sqrt(
        tf.reduce_sum(tf.math.squared_difference(vectorised_noise_a, vectorised_noise_b), 2))
    return distances_matrix


@tf.function
def loss_function_entropy(generated_noise, weights):
    dim_epsilon = tf.math.reduce_prod(generated_noise.shape) // generated_noise.shape[0]

    vect_epsilon = tf.reshape(generated_noise, (-1, dim_epsilon))

    dist_matrix = tf.sort(tf_euclidean_distance_matrix(vect_epsilon), axis=1)

    batch_size = dist_matrix.shape[0]

    k = int(tf.round(tf.math.sqrt(tf.cast(batch_size, tf.float32))))

    loss = -tf.math.reduce_sum(weights * tf.math.log(dist_matrix[:, k] + 1e-16)) * tf.cast(dim_epsilon,
                                                                                           tf.float32) / batch_size

    return loss


def entropy_estimation(weights, sigma, pos):
    weights_reshaped = tf.squeeze(weights)
    e_loss_det = weights_reshaped * tf.math.log(tf.linalg.det(sigma) ** 2 + 1e-12)
    e_loss_weights = weights_reshaped * tf.math.log(weights_reshaped + 1e-12)
    return -0.5 * tf.reduce_sum(e_loss_det) + tf.reduce_sum(e_loss_weights)


class Mixture(tf.keras.Model):
    def __init__(self, npart=100, nstart=64, nend=64, sigma=.1):
        """
        npart (int) : number of particles (Number of Gaussians in our Mixture)
        nstart (int) : dimension of input noise
        nend (int) : dimension of output noise
        sigma (int) : starting variance
        """
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.npart = npart
        self.nstart = nstart
        self.nend = nend
        self.logp = tf.Variable(1. + 0. * tf.random.normal(shape=(self.npart, 1), dtype="float32"))
        self.pos = tf.Variable(tf.random.normal(mean=0., shape=(self.npart, self.nstart), dtype="float32", stddev=3.),
                               name="pos")
        self.sigma = tf.Variable(sigma * tf.random.normal(shape=(self.npart, self.nend, self.nend)), dtype="float32")

    @tf.function
    def computeMixtureProb(self):
        p = tf.keras.activations.softmax(self.logp, axis=0)
        return p

    @tf.function
    def computeLowertTri(self):
        lowerTri = tf.Variable(np.zeros(shape=(self.npart, self.nstart, self.nstart)), dtype="float32")
        return lowerTri

    @tf.function
    def computeWeights(self, randind):
        p = self.computeMixtureProb()
        weights = tf.gather(p, randind)
        return weights  # /tf.reduce_sum(weights)

    # Corriger les poids en fonction du nombre dans chaque composante du mélange

    @tf.function
    def call(self, gaussian, vv=None, beta=1.):
        y = self.flatten(gaussian)
        vv = tf.transpose(self.logp)

        batch_size = gaussian.shape[0]

        # Sample random particles
        randind = tf.random.categorical(vv, num_samples=batch_size, dtype="int32")

        # Get the weights of our random particles
        weights = self.computeWeights(randind)

        # Get the mean of our random particles
        # displacement = (1-beta) * tf.random.normal(shape=(batch_size, self.nstart))
        pos = tf.reshape(tf.gather(self.pos, randind), (batch_size, self.nstart))  # + displacement

        # Multiply our variance by the input z -> Gives a Matrix
        temp = tf.matmul(self.sigma, tf.transpose(y))

        temp = tf.transpose(temp, (0, 2, 1))
        # Only select the particles we're interested in at this call
        ind = tf.transpose(tf.stack((randind[0, :], tf.range(batch_size))))
        vv = tf.gather_nd(params=temp, indices=ind, batch_dims=0)
        y = pos + vv
        y = tf.reshape(y, gaussian.shape)

        return y, tf.reshape(weights, (batch_size,)), randind


def train_inference_network(inference_network, generator, real_pixels, mask, history, epochs=4500, batch_size=150,
                            lr=0.01, optimiser=None):
    if optimiser is None:
        optimiser = tf.keras.optimizers.Adam(learning_rate=lr)
    infv = [inference_network.trainable_variables[i] for i in [0, 1, 2]]

    # gradients = None
    dim = inference_network.nstart

    logp_epoch = None

    conditionning_pixels_with_batch = tf.repeat(real_pixels, batch_size, axis=0)
    conditionning_pixels_vector = tf.reshape(conditionning_pixels_with_batch, (
    batch_size, real_pixels.shape[1] * real_pixels.shape[2] * real_pixels.shape[3], -1))
    idx_conditionning_data, _ = tf.experimental.numpy.nonzero(conditionning_pixels_vector[0])
    idx_conditionning_data, _ = tf.unique(idx_conditionning_data)
    idx_conditionning_data = tf.repeat(tf.expand_dims(idx_conditionning_data, axis=0), batch_size, axis=0)

    for epoch in tqdm(range(epochs)):
        # generate noise m
        gaussian = tf.random.normal(shape=(batch_size, 2, 4, 8, 1))
        beta = 1
        with tf.GradientTape() as tape:
            tape.watch(infv)
            z_infered, weights, randind = inference_network(gaussian, logp_epoch, training=True)

            # Entropy
            all_pos = inference_network.pos
            all_sigma = inference_network.sigma
            all_weights = inference_network.computeMixtureProb()
            entr = entropy_estimation(all_weights, all_sigma, all_pos)

            # Generator
            infered_gen_data = generator(z_infered)[-1]  # TODO HARDCODED
            likelihood_loss, prior_loss = loss_function_crossentropy(z_infered, infered_gen_data,
                                                                     conditionning_pixels_vector, mask,
                                                                     idx_conditionning_data, weights)
            total_loss = entr + likelihood_loss + prior_loss

            # retropropagate loss
            gradients = tape.gradient(total_loss, infv)
            optimiser.apply_gradients(zip(gradients, infv))

        # History
        t_loss_numpy = total_loss.numpy()
        c_loss_numpy = likelihood_loss.numpy()
        history["Total Loss"].append(t_loss_numpy)
        history["Cross Loss"].append(c_loss_numpy)

    return optimiser, gradients
