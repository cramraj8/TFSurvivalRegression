# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np


TFRECORD_FILE = "data.tfrecords"


def read_tfrecord(tfrecords_filename="data.tfrecords"):
    """The function for reading the tf.record file.

    This function takes input as the protobuff file name. Then, given the
    header strings along with their data-types, this function will
    retrieve those respective values from the byte file through a loop.
    Finaaly, it will convert those strings back to the desired data-type
    (here its np.asarray[dtype=np.float32]).

    Args:
        tfrecords_filename: This carries the tfrecord file-name with the file-path.

    Returns:
        predictors: A numpy array(M*N) contains 'np.float32' elements.
        This variable returns the feature values matrix.
        gnd_truths: A numpy array(M*1) contains 'np.float32' elements.
        This variable returns the observed survival values vector.

    """

    predictors = []
    gnd_truths = []
    record_iterator = tf.python_io.tf_record_iterator(path=TFRECORD_FILE)

    for element in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(element)

        predictor_string = (example.features.feature['predictor_string']
                            .bytes_list
                            .value[0])
        gnd_truth_string = (example.features.feature['gnd_truth_string']
                            .bytes_list
                            .value[0])

        predictor = np.fromstring(predictor_string, dtype=np.float32)
        gnd_truth = np.fromstring(gnd_truth_string, dtype=np.float32)

        print('predictor : ', predictor)
        print('gnd_truth : ', gnd_truth)

        predictors.append((predictor))
        gnd_truths.append((gnd_truth))

    return predictors, gnd_truths


if __name__ == '__main__':

    predictors, gnd_truths = read_tfrecord(TFRECORD_FILE)

    # print('predictors np.array : ', predictors)
    # print('gnd_truth np.array : ', gnd_truths)
