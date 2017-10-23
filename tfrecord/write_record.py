# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np


TFRECORD_FILE = "data.tfrecords"


def _Float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_tfrecord(tfrecords_filename="data.tfrecords",
                    predictors=np.asarray([[1.1, 2.2], [4.5, 3.3], [8.7, 6.7]]),
                    gnd_truths=np.asarray([1.2, 2.3, 3.1])):
    """The function for creating a tfrecord file.

    This function takes input as the protobuff file-name and the storable
    dataset values. Through a loop this function converts the input values into a
    serializable string format and saves into a protobuff format. This function
    stores the dataset in the filename specified by the input argument.

    Args:
        predictors: A numpy array(M*N) contains 'np.float32' elements.
        This variable contains the feature values matrix.
        gnd_truths: A numpy array(M*1) contains 'np.float32' elements.
        This variable contains the observed survival values vector.

    Returns:
        None

    """

    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    assert len(predictors) == len(gnd_truths), 'Input records length mismatch !!'
    for predictor, gnd_truth in zip(predictors, gnd_truths):
        predictor_string = predictor.tostring()
        gnd_truth_string = gnd_truth.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'predictor_string': _bytes_feature(tf.compat.as_bytes(predictor_string)),
            'gnd_truth_string': _bytes_feature(tf.compat.as_bytes(gnd_truth_string))
        }))
        writer.write(example.SerializeToString())

    writer.close()


if __name__ == '__main__':

    PREDICTORS = np.asarray([[1.1, 2.2], [4.5, 3.3], [8.7, 6.7]], dtype=np.float32)
    GND_TRUTHS = np.asarray([1.2, 2.3, 3.1], dtype=np.float32)
    create_tfrecord(TFRECORD_FILE, PREDICTORS, GND_TRUTHS)
