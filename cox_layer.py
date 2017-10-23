import tensorflow as tf
import numpy as np


def cost(prediction, at_risk_label, observed):
    """The function for calculating the loss interms of log partial likelihood.

    This function brings output vector from the deep neural network and the
    calculated survival risk vector. Then it calculates the log partial
    likelihood only for the observed patients.

    Args
    ----
        prediction : numpy foat32 array representing the output of DNN.
        at_risk_label : numpy folat32 representing the at_risk score.
        observed : numpy int32 representing non-censored patient status.

    Returns
    -------
        cost : numpy float32 scalar representing the cost calculated from the
        prediction by DNN.

    """

    n_observations = at_risk_label.shape[0]
    exp = tf.reverse(tf.exp(prediction), axis=[0])
    partial_sum_a = cumsum(exp, n_observations)
    partial_sum = tf.reverse(partial_sum_a, axis=[0]) + 1
    log_at_risk = tf.log(tf.gather(partial_sum, tf.reshape(at_risk_label, [-1])) + 1e-50)
    diff = prediction - log_at_risk
    times = tf.reshape(diff, [-1]) * observed
    cost = - (tf.reduce_sum(times))
    return cost


def cumsum(x, observations):
    """The function for calculating cumulative sumation vector.

    This function receives a vector input and calculates its cumulative function
    representation as another vector.

    Args
    ----
        x : numpy float32 representing the vector input
        observations : int  representing the length of x vector.

    Returns
    -------
        cumsum : numpy float32 vector representing the cumulated sum of the input.

    """

    x = tf.reshape(x, (1, observations))
    # values = tf.split(1, x.get_shape()[1], x)
    values = tf.split(x, x.get_shape()[1], 1)
    out = []
    prev = tf.zeros_like(values[0])
    for val in values:
        s = prev + val
        out.append(s)
        prev = s
    cumsum = tf.concat(out, 1)
    cumsum = tf.reshape(cumsum, (observations, 1))
    return cumsum


if __name__ == '__main__':
    cost()
