from deel.lip.activations import GroupSort
import tensorflow as tf

def normalized_swish(x):
    """
    Normalized version of the Swish Activation function to make it 1-Lipschitz (Normalized, Non-Monotonic, 1-Lipschitz bounded functions)
    Important for Wasserstein (see theorie in paper)
    Args:
        x: float

    Returns: float

    """
    return (1 / 1.09984) * tf.keras.activations.swish(x)

def maxsort(x):
    """
    MaxSort activation function. 1-Lipschitz activation functions
    Important for Wasserstein (see theorie in paper)
    Args:
        x: float

    Returns: float

    """
    return GroupSort(2)(x)
