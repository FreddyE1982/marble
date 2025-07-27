import numpy as np
import tensorflow as tf
from marble_core import _W1, _B1, _W2, _B2, Core, configure_representation_size


class MarbleKerasLayer(tf.keras.layers.Layer):
    """Keras layer mirroring Marble's message passing MLP."""

    def __init__(self, core: Core, **kwargs) -> None:
        super().__init__(**kwargs)
        self.core = core
        self.w1 = tf.Variable(_W1.astype(np.float32))
        self.b1 = tf.Variable(_B1.astype(np.float32))
        self.w2 = tf.Variable(_W2.astype(np.float32))
        self.b2 = tf.Variable(_B2.astype(np.float32))

    def call(self, inputs: tf.Tensor) -> tf.Tensor:  # pragma: no cover
        h = tf.tanh(tf.matmul(inputs, self.w1) + self.b1)
        return tf.tanh(tf.matmul(h, self.w2) + self.b2)


def core_to_tf(core: Core) -> MarbleKerasLayer:
    """Return a :class:`MarbleKerasLayer` for use in TensorFlow graphs."""
    return MarbleKerasLayer(core)


def tf_to_core(layer: MarbleKerasLayer, core: Core) -> None:
    """Update Marble's global MLP weights from ``layer``."""
    rep_size = int(layer.w1.shape[0])
    configure_representation_size(rep_size)
    _w1 = layer.w1.numpy().astype(np.float64)
    _b1 = layer.b1.numpy().astype(np.float64)
    _w2 = layer.w2.numpy().astype(np.float64)
    _b2 = layer.b2.numpy().astype(np.float64)
    globals_dict = globals()
    globals_dict['_W1'] = _w1
    globals_dict['_B1'] = _b1
    globals_dict['_W2'] = _w2
    globals_dict['_B2'] = _b2
    for n in core.neurons:
        if n.representation.shape != (rep_size,):
            n.representation = np.zeros(rep_size, dtype=float)
