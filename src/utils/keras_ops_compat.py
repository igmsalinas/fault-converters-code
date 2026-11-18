"""Small Keras compatibility facade for the TensorFlow 2.7 JetPack stack."""


def set_mixed_precision_policy(policy):
    """Set one policy through Keras 3 or NVIDIA TensorFlow/Keras 2.7.

    NVIDIA's standalone Keras 2.7 package exposes the policy classes but not
    ``set_global_policy``. Its bundled ``tf.keras`` does expose the setter and
    controls the same TensorFlow mixed-precision runtime.
    """
    import keras

    setter = getattr(keras.mixed_precision, "set_global_policy", None)
    if setter is not None:
        setter(policy)
        return

    import tensorflow as tf

    setter = getattr(tf.keras.mixed_precision, "set_global_policy", None)
    if setter is not None:
        setter(policy)
        return
    experimental = getattr(tf.keras.mixed_precision, "experimental", None)
    setter = getattr(experimental, "set_policy", None)
    if setter is None:
        raise RuntimeError("This TensorFlow/Keras stack cannot set a precision policy")
    setter(policy)

try:
    from keras import ops  # Keras 3
except ImportError:
    import tensorflow as tf

    class _TensorFlowOps:
        """TensorFlow equivalents used by this project's custom losses/trainer."""

        convert_to_tensor = staticmethod(tf.convert_to_tensor)
        shape = staticmethod(tf.shape)
        reshape = staticmethod(tf.reshape)
        mean = staticmethod(tf.reduce_mean)
        square = staticmethod(tf.square)
        abs = staticmethod(tf.abs)
        minimum = staticmethod(tf.minimum)
        maximum = staticmethod(tf.maximum)
        normalize = staticmethod(tf.math.l2_normalize)
        matmul = staticmethod(tf.matmul)
        transpose = staticmethod(tf.transpose)
        arange = staticmethod(tf.range)
        sum = staticmethod(tf.reduce_sum)
        expand_dims = staticmethod(tf.expand_dims)
        concatenate = staticmethod(tf.concat)
        zeros = staticmethod(tf.zeros)
        sqrt = staticmethod(tf.sqrt)
        cast = staticmethod(tf.cast)

    ops = _TensorFlowOps()


__all__ = ["ops", "set_mixed_precision_policy"]
