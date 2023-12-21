import tensorflow as tf
import numpy as np
from tqdm.auto import tqdm
from models.conditional_generation.conditional_losses import mixture_entropy_estimation, \
    loss_weighted_function_crossentropy_2d


class Mixture(tf.keras.Model):
    def __init__(self, npart=100, nstart=128, nend=128, sigma=.1):
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

    # Corriger les poids en fonction du nombre dans chaque composante du mélange ?

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
        pos = tf.reshape(tf.gather(self.pos, randind), (batch_size, self.nstart))

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
    gradients = None
    dim = inference_network.nstart
    logp_epoch = None

    conditionning_pixels_with_batch = tf.repeat(real_pixels, batch_size, axis=0)
    conditionning_pixels_vector = tf.reshape(conditionning_pixels_with_batch,
                                             (batch_size, real_pixels.shape[1] * real_pixels.shape[2], -1))
    idx_conditionning_data, _ = tf.experimental.numpy.nonzero(conditionning_pixels_vector[0])
    idx_conditionning_data, _ = tf.unique(idx_conditionning_data)
    idx_conditionning_data = tf.repeat(tf.expand_dims(idx_conditionning_data, axis=0), batch_size, axis=0)

    for epoch in tqdm(range(epochs)):
        # generate noise m
        gaussian = tf.random.normal(shape=(batch_size, 8, 16, 1))

        # beta = tf.math.sin(tf.math.minimum(1., 0.01+epoch/epochs) * math.pi/2)
        beta = 1
        with tf.GradientTape() as tape:
            tape.watch(infv)

            # m through the inference to generate z
            z_infered, weights, randind = inference_network(gaussian, logp_epoch, training=True)

            all_pos = inference_network.pos
            all_sigma = inference_network.sigma
            all_weights = inference_network.computeMixtureProb()
            entr = mixture_entropy_estimation(all_weights, all_sigma, all_pos)
            # Entropy

            # Generator
            infered_gen_data = generator(z_infered)[-1]  # TODO [-1] HARDCODED
            likelihood_loss, prior_loss = loss_weighted_function_crossentropy_2d(z_infered, infered_gen_data,
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
