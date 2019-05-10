import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler


def get_moons_dataset(n_samples=1000):
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    X, _ = noisy_moons
    X = StandardScaler().fit_transform(X)

    dataset = tf.data.Dataset.from_tensor_slices(X.astype(np.float32))
    return dataset

def get_mnist_dataset(hparams):

    def _convert_types(features):
        image = tf.cast(features["image"], tf.float32)
        image += tf.random.uniform(image.shape)
        image /= 256
        image = tf.reshape(image, [image.shape[0] * image.shape[1]])
        label = tf.one_hot(features["label"], depth=10)
        return label, image

    dataset, info = tfds.load('mnist', with_info=True)
    mnist_train, mnist_test = dataset['train'], dataset['test']
    mnist_train = mnist_train.shuffle(20000)
    mnist_train = mnist_train.apply(
        tf.data.experimental.map_and_batch(
            _convert_types,
            hparams.batch_size,
            num_parallel_calls=hparams.num_parallel_calls
        )
    )
    mnist_train = mnist_train.prefetch(2)

    mnist_test = mnist_test.shuffle(1000)
    mnist_test = mnist_test.apply(
        tf.data.experimental.map_and_batch(
            _convert_types,
            hparams.batch_size,
            num_parallel_calls=hparams.num_parallel_calls
        )
    )
    mnist_test = mnist_test.prefetch(2)

    train_iterator = mnist_train.make_initializable_iterator()

    return train_iterator

def get_rt_dataset(hparams, file_glob):

    feature_description = {
        'origin': tf.FixedLenFeature([3], tf.float32, default_value=[0, 0, 0]),
        'incident': tf.FixedLenFeature([3], tf.float32, default_value=[0, 0, 0]),
        'normal': tf.FixedLenFeature([3], tf.float32, default_value=[0, 0, 0]),
        'Li': tf.FixedLenFeature([3], tf.float32, default_value=[0, 0, 0]),
        'image_rgb': tf.FixedLenFeature([], tf.string),
        'image_depth': tf.FixedLenFeature([], tf.string),
        'image_position': tf.FixedLenFeature([], tf.string)
    }

    def _process(example):
        origin = example['origin']
        incident = example['incident']
        normal = example['normal']
        Li = example['Li'] + 1e-3

        image_rgb = tf.map_fn(
            lambda x: tf.image.decode_png(tf.read_file(x), channels=3),
            example["image_rgb"],
            dtype=tf.uint8
        )
        image_rgb = tf.cast(image_rgb, tf.float32) / 255
        # image_rgb = tf.reshape(image_rgb, shape=[tf.shape(image_rgb)[0], 256, 256, 3])

        image_depth = tf.map_fn(
            lambda x: tf.image.decode_png(tf.read_file(x), channels=1),
            example["image_depth"],
            dtype=tf.uint8
        )
        image_depth = tf.cast(image_depth, tf.float32) / 255
        # image_depth = tf.reshape(image_depth, shape=[tf.shape(image_depth)[0], 256, 256, 1])

        image_position = tf.map_fn(
            lambda x: tf.image.decode_png(tf.read_file(x), channels=3),
            example["image_position"],
            dtype=tf.uint8
        )
        image_position = tf.cast(image_position, tf.float32) / 255
        # image_position = tf.reshape(image_position, shape=[tf.shape(image_depth)[0], 256, 256, 3])
        # Li = example['Li'] * 255
        # Li += tf.random.uniform(tf.shape(Li), minval=0, maxval=1)
        # Li /= 256
        # alpha = 1e-3
        # Li = Li * (1-alpha) + alpha*0.5
        return origin, incident, normal, Li, image_rgb, image_depth, image_position

    filenames = tf.data.Dataset.list_files(file_glob)
    dataset = filenames.apply(tf.data.experimental.shuffle_and_repeat(10000, hparams.num_epochs))
    dataset = dataset.apply(tf.data.experimental.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=4))
    dataset = dataset.batch(hparams.batch_size)
    dataset = dataset.apply(
        tf.data.experimental.parse_example_dataset(
            feature_description,
            num_parallel_calls=hparams.num_parallel_calls
        )
    )
    dataset = dataset.map(
        _process, num_parallel_calls=hparams.num_parallel_calls)
    dataset = dataset.prefetch(2)

    iterator = dataset.make_initializable_iterator()
    return iterator
