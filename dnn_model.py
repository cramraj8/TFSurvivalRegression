# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf


slim = tf.contrib.slim


def multilayer_nn_model(inputs, hidden_layers, n_classes, beta,
                        scope="deep_regression"):
    """Creates a deep regression model.

    This function takes input as the required parameters to build a deep
    neural network and builds the layer-wise network. Once its instance is
    called by parsing the input feedings, this function will perform the
    feed-forward and returns the output layer responses.

    Args:
        inputs: A node that yields a `Tensor` of size [total_observations,
        input_features].

    Returns:
        predictions: `Tensor` of shape (1) (scalar) of response.
        end_points: A dict of end points representing the hidden layers.
    """

    end_points = {}
    with slim.arg_scope([slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(beta)):
        net = slim.stack(inputs,
                         slim.fully_connected,
                         hidden_layers,
                         scope='fc')
        end_points['fc'] = net
        predictions = slim.fully_connected(net, n_classes, activation_fn=None,
                                           scope='prediction')
        end_points['out'] = predictions
        return predictions, end_points
