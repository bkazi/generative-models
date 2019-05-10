from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import tensorflow_probability as tfp
import itertools
import matplotlib.pyplot as plt
from math import pi

from HParams import HParams
from Bijectors import Chain
from RealNVP import real_nvp_template

tfd = tfp.distributions
tfb = tfp.bijectors

print("Tensorflow Version {}".format(tf.VERSION))
print("Tensorflow Probability Version {}".format(tfp.__version__))

hparams = HParams(
    batch_size=100,
    n_couplings=10,
    learning_rate=1e-5,
    l2_coeff=1e0,
    clip_gradient=1e2,
    num_parallel_calls=12,
    num_epochs=int(1e3))

def pack_images(images, rows, cols):
    """Helper utility to make a field of images."""
    shape = tf.shape(images)
    width, height, depth = shape[-3], shape[-2], shape[-1]
    images = tf.reshape(images, (-1, width, height, depth))
    batch = tf.shape(images)[0]
    rows = tf.minimum(rows, batch)
    cols = tf.minimum(batch // rows, cols)
    images = images[:rows * cols]
    images = tf.reshape(images, (rows, cols, width, height, depth))
    images = tf.transpose(images, [0, 2, 1, 3, 4])
    images = tf.reshape(images, [1, rows * width, cols * height, depth])
    return images

def get_mnist_dataset(hparams):

    def _convert_types(features):
        image = tf.cast(features["image"], tf.float32)
        image += tf.random.uniform(image.shape)
        image /= 256

        rotation = tf.random.uniform([], minval=0, maxval=1.5*pi)
        image = tf.contrib.image.rotate(
            image,
            rotation
        )

        image = tf.reshape(image, [image.shape[0] * image.shape[1]])
        rotation = tf.reshape(rotation, [1])
        label = tf.one_hot(features["label"], depth=10)
        return label, rotation, image

    dataset, info = tfds.load('mnist', with_info=True)
    mnist_train, mnist_test = dataset['train'], dataset['test']
    mnist_train = mnist_train.apply(
        tf.data.experimental.shuffle_and_repeat(50000, hparams.num_epochs)
    )
    mnist_train = mnist_train.apply(
        tf.data.experimental.map_and_batch(
            _convert_types,
            hparams.batch_size,
            num_parallel_calls=hparams.num_parallel_calls
        )
    )
    mnist_train = mnist_train.prefetch(1)

    mnist_test = mnist_test.apply(
        tf.data.experimental.shuffle_and_repeat(50000, hparams.num_epochs)
    )
    mnist_test = mnist_test.apply(
        tf.data.experimental.map_and_batch(
            _convert_types,
            hparams.batch_size,
            num_parallel_calls=hparams.num_parallel_calls
        )
    )
    mnist_test = mnist_test.prefetch(1)

    train_iterator = mnist_train.make_initializable_iterator()
    test_iterator = mnist_test.make_initializable_iterator()

    return train_iterator, test_iterator

def g(x, D, hparams, training=True):
    _g = [64, 64, 64, 64]
    regularizer = tf.contrib.layers.l2_regularizer(scale=hparams.l2_coeff)
    initializer = tf.contrib.layers.xavier_initializer()

    for i, units in enumerate(_g):
        x = tf.layers.dense(
            inputs=x,
            units=units,
            kernel_regularizer=regularizer,
            kernel_initializer=initializer,
            activation=tf.nn.leaky_relu)
        if i % 2:
            x = tf.layers.dropout(
                x,
                rate=0.4,
                training=training
            )

    p_x = tf.layers.dense(
        inputs=x,
        units=2 * D,
        kernel_regularizer=regularizer,
        kernel_initializer=initializer,
        activation=None)

    def _e_x():
        return tf.layers.dense(
            inputs=x,
            units=128,
            kernel_regularizer=regularizer,
            kernel_initializer=initializer,
            activation=None)
    e_x = tf.make_template("extra_info", _e_x)

    return p_x, e_x

def model(D, d, hparams, training=True):
    with tf.name_scope("model"):
        def _model(x):
            regularizer = tf.contrib.layers.l2_regularizer(scale=hparams.l2_coeff)
            initializer = tf.contrib.layers.xavier_initializer()

            prior, extra_info = g(x, D, hparams, training)

            coupling_layers = []
            L = hparams.n_couplings
            for i in range(L):
                coupling_layers.append(tfb.RealNVP(
                    num_masked=d,
                    shift_and_log_scale_fn=real_nvp_template(
                        hidden_layers=[1024, 512, 256, 128, 128, 256, 512, 1024],
                        extra_info=extra_info,
                        regularizer=regularizer,
                        initializer=initializer,
                        training=training,
                        use_batch_norm=False,
                        use_drop_out=True
                    )
                ))
                if i % 2 == 0:
                    coupling_layers.append(tfb.BatchNormalization(training=training))
                if not (i + 1) == L:
                    coupling_layers.append(tfb.Permute(
                        permutation=list(itertools.chain(range(d, D), range(d)))))

            coupling_layers.append(tfb.Sigmoid())
            alpha = 1e-5
            coupling_layers.append(tfb.Invert(tfb.AffineScalar(scale=(1 - alpha), shift=(alpha * 0.5))))

            coupling_layers = list(reversed(coupling_layers))
            bijector = Chain(coupling_layers)

            loc, scale = tf.split(prior, 2, axis=1)
            scale = tf.nn.softplus(scale) + 1e-3
            # loc, scale = tf.zeros([tf.shape(x)[0], D], dtype=tf.float32), tf.ones([tf.shape(x)[0], D], dtype=tf.float32)
            theta = tfd.Independent(
                distribution=tfd.Logistic(loc=loc, scale=scale),
                reinterpreted_batch_ndims=1
            )
            dist = tfd.ConditionalTransformedDistribution(
                distribution=theta,
                bijector=bijector,
            )
            return dist

    return tf.make_template("model_template", _model)

D = 28*28
d = int(D // 2)

is_training = tf.placeholder_with_default(True, shape=[])
label = tf.placeholder(tf.float32, shape=[None, 10])
rotation = tf.placeholder(tf.float32, shape=[None, 1])
_model_fn = model(D, d, hparams, training=is_training)
x = tf.concat([label, rotation], axis=-1)
distribution = _model_fn(x)

samples = distribution.sample()
samples = tf.reshape(samples, [-1, 28, 28, 1])
samples = pack_images(samples, 5, 10)

_label = np.eye(10)[np.tile([0, 1, 2, 3, 4, 5, 6 ,7, 8, 9], 5)]
# _rotation = np.repeat([0, 0.5*pi, pi, 1.5*pi, 2*pi], 10).reshape((-1, 1))
_rotation = np.repeat([0, 0, 0, 0, 0], 10).reshape((-1, 1))

_, train_iterator = get_mnist_dataset(hparams)
_, _, _img, = train_iterator.get_next()
_img = tf.reshape(_img, [-1, 28, 28, 1])
_img = pack_images(_img, 10, 10)

checkpoint_prefix = './mnist-checkpoints/model.ckpt'
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_prefix)
    sess.run(train_iterator.initializer)
    sess.graph.finalize()

    # img_ = sess.run(_img)
    # img_ = img_.reshape((img_.shape[1], img_.shape[2]))
    # plt.imshow(img_, cmap='Greys')
    # plt.axis('off')
    # plt.show()

    samples_ = sess.run(
        samples,
        feed_dict={
            is_training: False,
            label: _label,
            rotation: _rotation
        }
    )

    samples_ = samples_.reshape((samples_.shape[1], samples_.shape[2]))
    plt.imshow(samples_, cmap='Greys')
    plt.axis('off')
    plt.show()
