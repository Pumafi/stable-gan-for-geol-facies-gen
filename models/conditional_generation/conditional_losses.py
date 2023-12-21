import tensorflow as tf


def loss_function_crossentropy_2d(generated_noise, generated_images, conditionning_pixels_vector, mask,
                                  index_conditionning_data):
    mask = tf.repeat(mask, generated_images.shape[0], axis=0)
    fake_pixels = tf.squeeze(tf.keras.layers.Multiply()([generated_images, mask]))

    fake_pixels_flat = tf.reshape(fake_pixels, (fake_pixels.shape[0], fake_pixels.shape[1] * fake_pixels.shape[2], -1))
    fake_pixels_cond_values = tf.gather(fake_pixels_flat, index_conditionning_data, batch_dims=1)

    real_pixels_cond_values = tf.gather(conditionning_pixels_vector, index_conditionning_data, batch_dims=1)

    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    neg_log_likelihood = cce(real_pixels_cond_values, fake_pixels_cond_values)

    neg_log_likelihood = tf.reduce_sum(neg_log_likelihood, axis=[1])

    prior = 0.5 * tf.reduce_sum(tf.square(generated_noise), axis=[1, 2])

    return tf.math.reduce_mean(neg_log_likelihood), tf.math.reduce_mean(prior)


def loss_function_crossentropy_3d(generated_noise, generated_images, conditionning_pixels_vector, mask,
                                  index_conditionning_data):
    """
    TODO: it's almost the same code as 2D, maybe can merge the codes ?
    """
    # l2
    mask = tf.repeat(mask, generated_images.shape[0], axis=0)
    fake_pixels = tf.squeeze(tf.keras.layers.Multiply()([generated_images, mask]))

    fake_pixels_flat = tf.reshape(fake_pixels, (
        fake_pixels.shape[0], fake_pixels.shape[1] * fake_pixels.shape[2] * fake_pixels.shape[3], -1))
    fake_pixels_cond_values = tf.gather(fake_pixels_flat, index_conditionning_data, batch_dims=1)
    real_pixels_cond_values = tf.gather(conditionning_pixels_vector, index_conditionning_data, batch_dims=1)

    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    neg_log_likelihood = cce(real_pixels_cond_values, fake_pixels_cond_values)
    neg_log_likelihood = tf.reduce_sum(neg_log_likelihood, axis=[1])

    prior = 0.5 * tf.reduce_sum(tf.square(generated_noise), axis=[1, 2, 3])

    return tf.math.reduce_mean(neg_log_likelihood), tf.math.reduce_mean(prior)


def loss_weighted_function_crossentropy_2d(generated_noise, generated_images, conditionning_pixels_vector, mask,
                                           index_conditionning_data, weights):
    mask = tf.repeat(mask, generated_images.shape[0], axis=0)
    fake_pixels = tf.squeeze(tf.keras.layers.Multiply()([generated_images, mask]))

    fake_pixels_flat = tf.reshape(fake_pixels, (fake_pixels.shape[0], fake_pixels.shape[1] * fake_pixels.shape[2], -1))
    fake_pixels_cond_values = tf.gather(fake_pixels_flat, index_conditionning_data, batch_dims=1)

    real_pixels_cond_values = tf.gather(conditionning_pixels_vector, index_conditionning_data, batch_dims=1)

    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    neg_log_likelihood = cce(real_pixels_cond_values, fake_pixels_cond_values)

    neg_log_likelihood = tf.reduce_sum(neg_log_likelihood, axis=[1])

    prior = 0.5 * tf.reduce_sum(tf.square(generated_noise), axis=[1, 2])

    return tf.math.reduce_mean(weights * tf.math.reduce_mean(neg_log_likelihood, axis=-1)), tf.math.reduce_mean(
        weights * prior)


def mixture_entropy_estimation(weights, sigma, pos):
    weights_reshaped = tf.squeeze(weights)
    e_loss_det = weights_reshaped * tf.math.log(tf.linalg.det(sigma) ** 2 + 1e-12)
    e_loss_weights = weights_reshaped * tf.math.log(weights_reshaped + 1e-12)
    return -0.5 * tf.reduce_sum(e_loss_det) + tf.reduce_sum(e_loss_weights)


@tf.function
def tf_euclidean_distance_matrix(vectorised_noise):
    vectorised_noise_a = tf.expand_dims(vectorised_noise, 1)
    vectorised_noise_b = tf.expand_dims(vectorised_noise, 0)
    distances_matrix = tf.math.sqrt(
        tf.reduce_sum(tf.math.squared_difference(vectorised_noise_a, vectorised_noise_b), 2))
    return distances_matrix


@tf.function
def kozachenko_entropy_estimation(generated_noise):
    dim_epsilon = tf.math.reduce_prod(generated_noise.shape) // generated_noise.shape[0]

    vect_epsilon = tf.reshape(generated_noise, (-1, dim_epsilon))

    dist_matrix = tf.sort(tf_euclidean_distance_matrix(vect_epsilon), axis=1)

    batch_size = dist_matrix.shape[0]

    k = int(tf.round(tf.math.sqrt(tf.cast(batch_size, tf.float32))))

    loss = -tf.math.reduce_sum(tf.math.log(dist_matrix[:, k] + 1e-16)) * tf.cast(dim_epsilon, tf.float32) / batch_size

    return loss
