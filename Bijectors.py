import tensorflow as tf
from tensorflow_probability import bijectors as tfb


class Chain(tfb.ConditionalBijector, tfb.Chain):
    pass
