"""
models/dann_layers.py
=====================
Gradient Reversal Layer (GRL) для DANN.

Прямой проход: identity
Обратный проход: умножение градиента на -alpha
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer


@tf.custom_gradient
def grad_reverse(x, alpha):
    """Прямой: identity. Обратный: dy * (-alpha)."""
    def grad(dy):
        return -alpha * dy
    return x, grad


class GradientReversalLayer(Layer):
    """
    Gradient Reversal Layer.

    alpha замыкается в Python-константу — обходит ошибку
    "returned too few gradients" в tf.custom_gradient.
    """

    def __init__(self, alpha=0.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def call(self, x):
        alpha = float(self.alpha)

        @tf.custom_gradient
        def _grad_reverse(x):
            def grad(dy):
                return -alpha * dy
            return x, grad

        return _grad_reverse(x)

    def set_alpha(self, alpha):
        self.alpha = alpha

    def get_config(self):
        config = super().get_config()
        config.update({'alpha': self.alpha})
        return config