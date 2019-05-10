import tensorflow as tf
import tensorflow_probability as tfp
import itertools

from RealNVP import real_nvp_template
from Bijectors import Chain

tfd = tfp.distributions
tfb = tfp.bijectors

def resnet_bottleneck(inp, filters, kernel_size, initializer=None, regularizer=None, training=True):
    image = inp
    final_filters = inp.shape.as_list()[-1]
    image = tf.layers.conv2d(
        image,
        filters[0],
        (1, 1),
        padding="same",
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        activation=tf.nn.leaky_relu
    )
    image = tf.layers.conv2d(
        image,
        filters[1],
        kernel_size,
        padding="same",
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        activation=tf.nn.leaky_relu
    )
    image = tf.layers.conv2d(
        image,
        final_filters,
        (1, 1),
        padding="same",
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        activation=tf.nn.leaky_relu
    )
    image = image + inp
    image = tf.layers.batch_normalization(
        image,
        gamma_initializer=initializer,
        beta_initializer=initializer,
        training=training,
    )
    image = tf.nn.leaky_relu(image)
    return image


def g(x, images, D, hparams, training=True):
    img_shape = tf.shape(images)
    img_shape = [-1, img_shape[1], img_shape[2], 1]

    _g = [32, 32, 32, 32, 32, 32]
    regularizer = tf.contrib.layers.l2_regularizer(scale=hparams.l2_coeff)
    initializer = tf.contrib.layers.xavier_initializer()

    _x = x
    for units in _g:
        _x = tf.layers.dense(
            inputs=_x,
            units=units,
            kernel_regularizer=regularizer,
            kernel_initializer=initializer,
            activation=tf.nn.leaky_relu)

    for units in [64]:
        _x = tf.layers.dense(
            inputs=_x,
            units=units,
            kernel_regularizer=regularizer,
            kernel_initializer=initializer,
            activation=tf.nn.leaky_relu)
    # img_mask = tf.layers.dense(
    #     inputs=_x,
    #     units=256 * 256,
    #     kernel_regularizer=regularizer,
    #     kernel_initializer=initializer,
    #     activation=tf.nn.sigmoid)
    # img_mask = tf.reshape(img_mask, shape=img_shape)

    # images = tf.concat([images, img_mask], axis=-1)
    images = tf.layers.conv2d(
        images,
        128,
        (1, 1),
        padding="same",
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        activation=tf.nn.leaky_relu
    )
    images = tf.layers.conv2d(
        images,
        128,
        (5, 5),
        padding="same",
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        activation=None
    )
    images = tf.layers.batch_normalization(
        images,
        gamma_initializer=initializer,
        beta_initializer=initializer,
        training=training,
    )
    images = tf.nn.leaky_relu(images)
    images = tf.layers.max_pooling2d(
        images,
        pool_size=(2, 2),
        padding="valid",
        strides=(2, 2)
    )
    images = tf.layers.conv2d(
        images,
        64,
        (3, 3),
        padding="same",
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        activation=None
    )
    images = tf.layers.batch_normalization(
        images,
        gamma_initializer=initializer,
        beta_initializer=initializer,
        training=training,
    )
    images = tf.nn.leaky_relu(images)
    images = tf.layers.max_pooling2d(
        images,
        pool_size=(2, 2),
        padding="valid",
        strides=(2, 2)
    )
    images = tf.layers.conv2d(
        images,
        32,
        (3, 3),
        padding="same",
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        activation=tf.nn.leaky_relu
    )
    # Global average pooling
    features = tf.reduce_mean(images, axis=[1,2])
    features = tf.contrib.layers.flatten(features)

    p_x = tf.layers.dense(
        inputs=_x,
        units=2 * D,
        kernel_regularizer=regularizer,
        kernel_initializer=initializer,
        activation=None)

    def _e_x():
        return tf.layers.dense(
            inputs=features,
            units=32,
            kernel_regularizer=regularizer,
            kernel_initializer=initializer,
            activation=tf.nn.leaky_relu)
        # e_x = tf.layers.dense(
        #     inputs=e_x,
        #     units=32,
        #     kernel_regularizer=regularizer,
        #     kernel_initializer=initializer,
        #     activation=None)
    e_x = tf.make_template("extra_info", _e_x)

    return p_x, e_x

def model(D, d, hparams, training=True):
    with tf.name_scope("model"):
        def _model(x, images):
            regularizer = tf.contrib.layers.l2_regularizer(scale=hparams.l2_coeff)
            initializer = tf.contrib.layers.xavier_initializer()

            prior, extra_info = g(x, images, D, hparams, training)

            coupling_layers = []
            L = hparams.n_couplings
            for i in range(L):
                coupling_layers.append(tfb.RealNVP(
                    num_masked=d,
                    shift_and_log_scale_fn=real_nvp_template(
                        hidden_layers=[64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
                        extra_info=extra_info,
                        regularizer=regularizer,
                        initializer=initializer,
                        training=training
                    )
                ))
                if (i + 0) % 3 == 0:
                    coupling_layers.append(tfb.BatchNormalization(training=training))
                if not (i + 1) == L:
                    coupling_layers.append(tfb.Permute(
                        permutation=list(itertools.chain(range(d, D), range(d)))))

            coupling_layers.append(tfb.Softplus())

            coupling_layers = list(reversed(coupling_layers))
            bijector = Chain(coupling_layers)

            loc, scale = tf.split(prior, 2, axis=1)
            scale = tf.nn.softplus(scale) + 1e-3
    #       _, scale = tf.zeros([tf.shape(x)[0], D], dtype=tf.float32), tf.ones([tf.shape(x)[0], D], dtype=tf.float32)
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