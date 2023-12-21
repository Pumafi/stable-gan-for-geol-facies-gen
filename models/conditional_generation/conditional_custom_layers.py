import tensorflow as tf


class InvertLinear(tf.keras.layers.Layer):
    def __init__(self, output_shape, activation):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(1, activation=activation)
        self.dense2 = tf.keras.layers.Dense(output_shape, activation=None, use_bias=False)

    def call(self, inputs):
        return inputs + self.dense2(self.dense1(inputs))
