from __future__ import absolute_import, division, print_function

import tensorflow as tf
import pandas as pd
import numpy as np
from multiprocessing import Pool
import time

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  """Returns an bytes_list from bytes string"""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(origin, incident, normal, Li, rgb, depth, position):
  feature = {
      "origin": _float_feature(origin),
      "incident": _float_feature(incident),
      "normal": _float_feature(normal),
      "Li": _float_feature(Li),
      "image_rgb": _bytes_feature(rgb),
      "image_depth": _bytes_feature(depth),
      "image_position": _bytes_feature(position)
  }
  
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

def _process_and_write(fname, dataframe, image):
    with tf.python_io.TFRecordWriter(fname) as writer:
        for idx, row in dataframe.iterrows():
            e = serialize_example(
                [row['originX'], row['originY'], row['originZ']],
                [row['incidentX'], row['incidentY'], row['incidentZ']],
                [row['normalX'], row['normalY'], row['normalZ']],
                [row['LiR'], row['LiG'], row['LiB']],
                image["color"],
                image["depth"],
                image["position"]
            )
            writer.write(e)

def write_tfrecord(dataframe, filename_prefix, num_splits=1):
    if num_splits <= 0:
        raise ValueError("num_splits must be greater than 0")

    image = {
        "color": "",
        "depth": "",
        "position": "",
    }

    prefix = "./images/cornell-box-{}.png"
    for key in image.keys():
        image[key] = prefix.format(key).encode('utf-8')

    pool = Pool()
    splits = np.array_split(range(dataframe.shape[0]), num_splits)
    inputs = [
        ("{}-{}.tfrecord".format(filename_prefix, i), dataframe.iloc[split, :], image)
        for i, split in enumerate(splits)
    ]
    try:
        pool.starmap(_process_and_write, inputs)
    finally:
        pool.close()
        pool.join()

def main():
    start_time = time.time()
    df = pd.read_csv('./data/trainData.csv')

    cleared_df = df.query('LiR + LiG + LiB < 5')
    cleared_df_1 = cleared_df.query('LiR + LiG + LiB > 1.5')
    shuffled_df_1 = cleared_df_1.sample(frac=0.7)

    cleared_df_2 = cleared_df.query('LiR + LiG + LiB < 1.5')
    shuffled_df_2 = cleared_df_2.sample(frac=0.15)

    shuffled_df = pd.concat([shuffled_df_1, shuffled_df_2])

    print("Read CSV data ({:.2f}s)".format((time.time() - start_time)))

    train_df = shuffled_df
    # test_df = df.drop(index=train_df.index)

    print("Writing training TFRecord")
    start_time = time.time()
    filename = './data/traindata'
    write_tfrecord(train_df, filename, num_splits=10)
    print("Finished writing training TFRecord ({:.2f}s)".format((time.time() - start_time)))
    print("Count: {}".format(train_df.shape[0]))

    # print("Writing testing TFRecord")
    # start_time = time.time()
    # filename = './data/testdata'
    # write_tfrecord(test_df, filename, num_splits=2)
    # print("Finished writing testing TFRecord ({:.2f}s)".format((time.time() - start_time)))
    # print("Count: {}".format(test_df.shape[0]))

if __name__ == "__main__":
    main()