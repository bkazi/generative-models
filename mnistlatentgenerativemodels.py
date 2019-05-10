from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import itertools
import numpy as np
from math import pi
from tqdm import trange
from time import time

from RealNVP import real_nvp_template
from HParams import HParams
from Bijectors import Chain

tfd = tfp.distributions
tfb = tfp.bijectors

print("Tensorflow Version {}".format(tf.VERSION))
print("Tensorflow Probability Version {}".format(tfp.__version__))

hparams = HParams(
    batch_size=1024,
    n_couplings=10,
    learning_rate=1e-5,
    l2_coeff=1e0,
    clip_gradient=1e2,
    num_parallel_calls=12,
    num_epochs=int(5e2))

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

        # samples = tf.random.multinomial([[1., 1., 1., 1., 1.]], num_samples=1)
        # elems = tf.convert_to_tensor([0., 0.5*pi, 0.75*pi, pi, 1.5*pi])
        # rotation = elems[tf.cast(samples[0][0], tf.int32)]
        rotation = 0.
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

TRAIN_TOTAL_SIZE = 60000
TEST_TOTAL_SIZE = 10000

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

train_iterator, test_iterator = get_mnist_dataset(hparams)

steps_in_epoch = TRAIN_TOTAL_SIZE // hparams.batch_size
t_steps_in_epoch = TEST_TOTAL_SIZE // hparams.batch_size

D = 28*28
d = D // 2

LOG_FREQUENCY = 10

decay_steps = 100 * steps_in_epoch

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(
    hparams.learning_rate, global_step, decay_steps, 0.95, staircase=True)
lr_summary = tf.summary.scalar('learning_rate', learning_rate)
optimizer = tf.train.AdamOptimizer(learning_rate)

is_training = tf.placeholder_with_default(True, shape=[])
_model_fn = model(D, d, hparams, training=is_training)
label, rotation, image = train_iterator.get_next()
x = tf.concat([label, rotation], axis=-1)
distribution = _model_fn(x)

samples = distribution.sample()
samples = samples[:9, ...]
samples = tf.reshape(samples, [9, 28, 28, 1])
img_summary = tf.summary.image('samples', pack_images(samples, 3, 3), max_outputs=1)
nll = -tf.reduce_mean(distribution.log_prob(image))
reg_loss = tf.losses.get_regularization_loss()
loss = nll + reg_loss

# t_label, t_rotation, t_image = test_iterator.get_next()
# t_x = tf.concat([t_label, tf.cast(t_rotation, tf.float32)], axis=-1)
# t_distribution = _model_fn(t_x)

# t_samples = t_distribution.sample()
# t_samples = t_samples[:9, ...]
# t_samples = tf.reshape(t_samples, [9, 28, 28, 1])
# t_img_summary = tf.summary.image('val samples', pack_images(t_samples, 3, 3), max_outputs=1)

# t_nll = -tf.reduce_mean(t_distribution.log_prob(t_image))
# t_loss = t_nll

loss_summary = tf.summary.scalar('loss', nll)
# t_loss_summary = tf.summary.scalar('val loss', t_loss)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    # capped_gradients = gradients
    capped_gradients, gradient_norm = tf.clip_by_global_norm(
        gradients, clip_norm=hparams.clip_gradient)
    capped_gradients_and_variables = zip(capped_gradients, variables)
    train_op = optimizer.apply_gradients(
        capped_gradients_and_variables, global_step=global_step)

merged = tf.summary.merge([loss_summary, lr_summary, img_summary])
# t_merged = tf.summary.merge([t_loss_summary, t_img_summary])

init_op = tf.group(
    tf.global_variables_initializer(),
    tf.local_variables_initializer(),
    test_iterator.initializer,
    train_iterator.initializer
)

losses = []
log_dir = "./mnist-logs"
checkpoint_prefix = './mnist-checkpoints/model.ckpt'
with tf.Session() as sess:
    saver = tf.train.Saver()
    time_name = int(time())
    # train_file_writer = tf.summary.FileWriter(log_dir + "/train-{}".format(time_name), sess.graph)
    # test_file_writer = tf.summary.FileWriter(log_dir + "/test-{}".format(time_name), sess.graph)
    sess.run(init_op)
    sess.graph.finalize()

    for i in range(hparams.num_epochs):
        total_loss = 0

        t = trange(steps_in_epoch, desc='Epoch {}'.format(i + 1), unit="batches")
        for step in t:
            _, loss_ = sess.run([train_op, nll])
            total_loss += loss_
            t.set_description("Epoch {}: Loss {:.3f}".format(i, total_loss / (step + 1)))
            t.refresh()

        if i % LOG_FREQUENCY == 0 or i == hparams.num_epochs - 1:
            # t_total_loss = 0
            # for step in range(t_steps_in_epoch):
            #     t_loss_, t_summary_ = sess.run([t_loss, t_merged], feed_dict={is_training: False})
            #     t_total_loss += t_loss_

            saver.save(sess, checkpoint_prefix)
            # train_file_writer.add_summary(summary_, i)
            # test_file_writer.add_summary(t_summary_, i)
            avg_loss = total_loss / steps_in_epoch
            losses.append(avg_loss)
            # t_avg_loss = t_total_loss / t_steps_in_epoch
            # t.set_description("Epoch {}: Loss {:.3f} Val. Loss {:.3f}".format(i, avg_loss, t_avg_loss))
            # t.refresh()
            # print("Epoch {}: Loss {:.3f} Val. Loss {:.3f}".format(i, avg_loss, t_avg_loss))

losses = np.array(losses)
np.save("./out-loss", losses)