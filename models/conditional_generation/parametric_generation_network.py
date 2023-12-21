import tensorflow as tf
import tqdm
from utils.utils import correct_percentage
from models.conditional_generation.conditional_losses import kozachenko_entropy_estimation, loss_function_crossentropy_2d
from models.conditional_generation.conditional_custom_layers import InvertLinear
import numpy as np


class InferenceModel(tf.keras.Model):
    """
    Implementation of a sampling network.
    if nhidden is superior to 1, it will use InvertibleLinear layers; just like gradient flow
    if nhiddnen is 0 and the activation is set to swish or gelu; it will be the vanilla network used during most
    of the thesis and used for published results
    """

    def __init__(self, output_shape, nhidden=0, hidden_neurons_nb=8, hidden_activation=tf.keras.activations.swish,
                 final_activation=None):
        super().__init__()

        final_neurons_nb = np.prod(output_shape)
        self.out_shape = output_shape

        self.flat = tf.keras.layers.Flatten()
        self.listInvert = []
        for i in range(nhidden):
            self.listInvert += [InvertLinear(hidden_neurons_nb, hidden_activation)]
        self.denseF1 = tf.keras.layers.Dense(hidden_neurons_nb, activation=hidden_activation)
        self.denseF2 = tf.keras.layers.Dense(final_neurons_nb, activation=final_activation)

    def call(self, inputs):
        y = self.flat(inputs)
        for l in self.listInvert:
            y = l(y)
        y = self.denseF1(y)
        y = self.denseF2(y)
        y = tf.reshape(y, (-1, *self.out_shape))
        return y


def train_inference_network(inference_network, generator_network, real_pixels, mask, input_shape, epochs=4500,
                            batch_size=150, learning_rate=0.001, use_annealing=False, compute_success_rate=True):
    history = {"Total Loss": [], "Likelihood Loss": [], "Prior Loss": [], "Cross Entropy": [], "Negative Entropy": [],
               "Correct Percentage": []}
    optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    conditionning_pixels_with_batch = tf.repeat(real_pixels, batch_size, axis=0)
    conditionning_pixels_vector = tf.reshape(conditionning_pixels_with_batch,
                                             (batch_size, real_pixels.shape[1] * real_pixels.shape[2], -1))
    idx_conditionning_data, _ = tf.experimental.numpy.nonzero(conditionning_pixels_vector[0])
    idx_conditionning_data, _ = tf.unique(idx_conditionning_data)
    idx_conditionning_data = tf.repeat(tf.expand_dims(idx_conditionning_data, axis=0), batch_size, axis=0)
    for epoch in tqdm.tqdm(range(epochs)):

        if use_annealing:
            beta = min(1.0, 0.01 + epoch / 4500)
        else:
            beta = 1.0

        if epoch % 5000 == 0:
            # Learning rate decay
            if epoch != 0:
                optimiser.learning_rate = optimiser.learning_rate / 5
            print(optimiser.learning_rate)

        z_original = tf.random.normal(shape=(batch_size, *input_shape))
        with tf.GradientTape() as tape:
            # m through the inference to generate z
            z_infered = inference_network(z_original)

            # input z to gen network
            generated_images = generator_network(z_infered)[-1]

            likelihood_loss, prior_loss = loss_function_crossentropy_2d(z_infered, generated_images,
                                                                        conditionning_pixels_vector, mask,
                                                                        idx_conditionning_data)
            loss_entropy = kozachenko_entropy_estimation(z_infered)
            total_loss = loss_entropy + beta * (likelihood_loss + prior_loss)

        # retropropagate loss
        gradients = tape.gradient(total_loss, inference_network.trainable_variables)
        optimiser.apply_gradients(zip(gradients, inference_network.trainable_variables))

        # History
        t_loss_numpy = total_loss.numpy()
        history["Total Loss"].append(t_loss_numpy)

        likelihood_loss_numpy = likelihood_loss.numpy()
        history["Likelihood Loss"].append(likelihood_loss_numpy)
        prior_loss_numpy = prior_loss.numpy()
        history["Prior Loss"].append(prior_loss_numpy)

        loss_crossentropy_numpy = likelihood_loss.numpy() + prior_loss.numpy()
        history["Cross Entropy"].append(loss_crossentropy_numpy)
        loss_entropy_numpy = loss_entropy.numpy()
        history["Negative Entropy"].append(loss_entropy_numpy)
        if compute_success_rate:
            corr_percentage = correct_percentage(generated_images, real_pixels, mask)
            history["Correct Percentage"].append(corr_percentage)

    return history
