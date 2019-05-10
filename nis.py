from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow_probability as tfp
from math import ceil, floor
from tqdm import trange

from datasets import get_rt_dataset
from HParams import HParams
from Model import model

print("Tensorflow Version {}".format(tf.VERSION))
print("Tensorflow Probability Version {}".format(tfp.__version__))

TRAIN_TOTAL_SIZE = 641745
# TEST_TOTAL_SIZE = 41721

hparams = HParams(
    batch_size=32,
    n_couplings=6,
    learning_rate=3e-4,
    l2_coeff=2e-3,
    clip_gradient=1e2,
    num_parallel_calls=12,
    num_epochs=int(1e1))

file_glob = './data/traindata-*.tfrecord'
train_iterator = get_rt_dataset(hparams, file_glob)

file_glob = './data/testdata-*.tfrecord'
# test_iterator = get_rt_dataset(hparams, file_glob)

def nll_loss(iterator, model_fn):
    _origin, _incident, _normal, _Li, _image_rgb, _image_depth, _image_position  = iterator.get_next()
    _x = tf.concat([_origin, _incident, _normal], axis=1)
    _images = tf.concat([_image_rgb, _image_depth, _image_position], axis=-1)
    _distribution = model_fn(_x, images)
    neg_log_loss = -tf.reduce_mean(_distribution.log_prob(_Li))
    return neg_log_loss

D = 3
d = ceil(D // 2)

_model_fn = model(D, d, hparams)

LOG_FREQUENCY = 1e0

steps_in_epoch = floor(TRAIN_TOTAL_SIZE / hparams.batch_size)
# t_steps_in_epoch = floor(TEST_TOTAL_SIZE / hparams.batch_size)
decay_steps = int(5e4)

global_step = tf.train.get_or_create_global_step()
learning_rate = tf.train.exponential_decay(
    hparams.learning_rate, global_step, decay_steps, 0.95)
lr_summary = tf.summary.scalar('learning_rate', learning_rate)
optimizer = tf.train.AdamOptimizer(learning_rate)

_origin, _incident, _normal, _Li, _image_rgb, _image_depth, _image_position = train_iterator.get_next()
x = tf.concat([_origin, _incident, _normal], axis=1)
images = tf.concat([_image_rgb, _image_depth, _image_position], axis=-1)
distribution = _model_fn(x, images)
nll = -tf.reduce_mean(distribution.log_prob(_Li))
l2_loss = tf.losses.get_regularization_loss()
loss = nll + l2_loss
# t_loss = nll_loss(test_iterator, _model_fn)
loss_summary = tf.summary.scalar('loss', nll)
# t_loss_summary = tf.summary.scalar('val loss', t_loss)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    # capped_gradients = gradients
    capped_gradients, gradients_norm = tf.clip_by_global_norm(
        gradients, clip_norm=hparams.clip_gradient)
    capped_gradients_and_variables = zip(capped_gradients, variables)
    train_op = optimizer.apply_gradients(
        capped_gradients_and_variables, global_step=global_step)

merged = tf.summary.merge([loss_summary, lr_summary])
# t_merged = tf.summary.merge([t_loss_summary])

init_op = tf.group(
    tf.global_variables_initializer(),
    tf.local_variables_initializer(),
    train_iterator.initializer,
    # test_iterator.initializer
)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

inference_graph = tf.Graph()
with inference_graph.as_default():
    inf_model_fn = model(D, d, hparams, training=False)
    inf_origin = tf.placeholder(tf.float32, shape=[None, 3], name="origin")
    inf_incident = tf.placeholder(tf.float32, shape=[None, 3], name="incident")
    inf_normal = tf.placeholder(tf.float32, shape=[None, 3], name="normal")

    inf_image_rgb = tf.placeholder(tf.string, name="image_rgb")
    inf_image_rgb = tf.image.decode_png(tf.read_file(inf_image_rgb), channels=3)
    inf_image_rgb = tf.expand_dims(inf_image_rgb, 0)
    inf_image_rgb = tf.cast(inf_image_rgb, tf.float32) / 255
    # inf_image_rgb = tf.reshape(inf_image_rgb, shape=[1, 256, 256, 3])

    inf_image_depth = tf.placeholder(tf.string, name="image_depth")
    inf_image_depth = tf.image.decode_png(tf.read_file(inf_image_depth), channels=1)
    inf_image_depth = tf.expand_dims(inf_image_depth, 0)
    inf_image_depth = tf.cast(inf_image_depth, tf.float32) / 255
    # inf_image_depth = tf.reshape(inf_image_depth, shape=[1, 256, 256, 1])

    inf_image_position = tf.placeholder(tf.string, name="image_position")
    inf_image_position = tf.image.decode_png(tf.read_file(inf_image_position), channels=3)
    inf_image_position = tf.expand_dims(inf_image_position, 0)
    inf_image_position = tf.cast(inf_image_position, tf.float32) / 255
    # inf_image_position = tf.reshape(inf_image_position, shape=[1, 256, 256, 3])

    inf_images = tf.concat([inf_image_rgb, inf_image_depth, inf_image_position], axis=-1)
    inf_x = tf.concat([inf_origin, inf_incident, inf_normal], axis=1)
    inf_distribution = inf_model_fn(inf_x, inf_images)

    _sample = inf_distribution.sample()
    print(_sample)

    _prob = inf_distribution.prob(_sample)
    out = tf.divide(_sample, _prob, name="out")


log_dir = "./logs"
save_model_dir = "./model"
checkpoint_prefix = './checkpoints/model.ckpt'
tf.train.write_graph(
    inference_graph.as_graph_def(),
    save_model_dir,
    "model.pb",
    as_text=False
)

with tf.Session(config=config) as sess:
    train_file_writer = tf.summary.FileWriter(log_dir + "/train", sess.graph)
    test_file_writer = tf.summary.FileWriter(log_dir + "/test", sess.graph)
    saver = tf.train.Saver()

    sess.run(init_op)
    sess.graph.finalize()

    for i in range(hparams.num_epochs):
        total_loss = 0

        t = trange(steps_in_epoch, desc='Epoch {}'.format(i + 1), unit="batches")
        for step in t:
            _, loss_, summary_ = sess.run([train_op, nll, merged])
            total_loss += loss_
            t.set_description("Epoch {}: Loss {:.3f}".format(i, total_loss / (step + 1)))
            t.refresh()

            if step % int(1e3) == 0 == 0:
                saver.save(sess, checkpoint_prefix)

        if i % LOG_FREQUENCY == 0 or i == hparams.num_epochs - 1:
            # t_total_loss = 0
            # for step in range(t_steps_in_epoch):
            #     t_loss_, t_summary_ = sess.run([t_loss, t_merged])
            #     t_total_loss += t_loss_

            train_file_writer.add_summary(summary_, i)
            # test_file_writer.add_summary(t_summary_, i)
            saver.save(sess, checkpoint_prefix)
            avg_loss = total_loss / steps_in_epoch
            # t_avg_loss = t_total_loss / t_steps_in_epoch
            # t.set_description("Epoch {}: Loss {:.3f} Val. Loss {:.3f}".format(i, avg_loss, t_avg_loss))
            # t.refresh()
            # print("Epoch {}: Loss {:.3f} Val. Loss {:.3f}".format(i, avg_loss, t_avg_loss))
