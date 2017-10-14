# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import data_providers
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.framework import ops
slim = tf.contrib.slim


def run_survivalnet(user_params, beta=0.01, dropout_rate=0.6,
                    hidden_layers=[500, 500, 500],
                    init_lr=0.0002):
    """The function for model training.

    This function receives user_provided_params and hyper_parameter values
    as function-arguments. Then it creates the tensorflow model-graph, defines
    loss and optimizer. Finally creates a training loop and saves the results
    and logs in the sub-directory.

    Args:
        user_params: This is a python-dict data-type, containing user-changable paramers.
        beta: A float value represents regularizing constant.
        dropout_rate: A 'float' value represents drop out ratio.
        hidden_layers: A python-list with 'int' elements, each represents layer width.
        init_lr: A 'float' value represents 'initial learning rate'.

    Returns:
        None

    """

    #Extracting user_given_parameters from function argument
    n_epochs = user_params['n_epochs']
    split_ratio = user_params['split_ratio']
    batch_size = user_params['batch_size']
    lr_decay_rate = user_params['lr_decay_rate']
    n_classes = user_params['n_classes']
    data_file = user_params['data_file']
    ckpt_dir = user_params['ckpt_dir']

    data_x, data_y = data_providers.data_providers(data_file)
    data_x, data_y = shuffle(data_x, data_y, random_state=1)
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y,
                                                        test_size=split_ratio,
                                                        random_state=420)

    n_obs = x_train.shape[0]
    n_features = x_train.shape[1]

    #TF-Graph creation
    tf.reset_default_graph()
    with tf.Graph().as_default() as graph:
        logging.set_verbosity(tf.logging.INFO)

        if not tf.gfile.Exists(ckpt_dir):
            tf.gfile.MakeDirs(ckpt_dir)

        n_batches = n_obs / batch_size
        decay_steps = int(n_epochs * n_batches)

        #Create the model
        def multilayer_nn_model(inputs, hidden_layers, n_classes, beta,
                                scope="deep_regression_model"):

            with tf.variable_scope(scope, 'deep_regression', [inputs]):
                end_points = {}
                with slim.arg_scope([slim.fully_connected],
                                    activation_fn=tf.nn.relu,
                                    weights_regularizer=slim.l2_regularizer(beta)):
                    net = slim.stack(inputs,
                                     slim.fully_connected,
                                     hidden_layers,
                                     scope='fc')
                    end_points['fc'] = net
                    predictions = slim.fully_connected(net, n_classes,
                                                       activation_fn=None,
                                                       scope='prediction')
                    end_points['out'] = predictions
                    return predictions, end_points

        #Predict the values from deep neural network.
        pred, end_points = multilayer_nn_model(x_train, hidden_layers,
                                               n_classes, beta)

        global_step = get_or_create_global_step()

        lr = tf.train.exponential_decay(learning_rate=init_lr,
                                        global_step=global_step,
                                        decay_steps=decay_steps,
                                        decay_rate=lr_decay_rate,
                                        staircase=True)

        loss = tf.losses.mean_squared_error(tf.squeeze(pred), y_train)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)

        #Create slim-train model
        train_op = slim.learning.create_train_op(
            loss,
            optimizer,
            clip_gradient_norm=4,   # Gives quick convergence
            check_numerics=True,
            summarize_gradients=True)

        #Create the slim-training-loop
        final = slim.learning.train(
            train_op,
            ckpt_dir,
            log_every_n_steps=1,
            graph=graph,
            global_step=global_step,
            number_of_steps=n_epochs,
            save_summaries_secs=20,
            startup_delay_steps=0,
            saver=None,
            save_interval_secs=10,
            trace_every_n_steps=1
        )


if __name__ == '__main__':

    user_params = {
        'n_epochs': 500,
        'split_ratio': 0.2,
        'batch_size': 100,
        'lr_decay_rate': 0.9,
        'n_classes': 1,
        'data_file': ['./Brain_Integ_X.csv', './Brain_Integ_Y.csv'],
        'ckpt_dir': './ckpt_dir/'
    }

    hyper_params = {
        'beta': 0.01,
        'dropout_rate': 0.6,
        'hidden_layers': [500, 500, 500],
        'init_lr': 0.001
    }

    run_survivalnet(user_params, **hyper_params)
    #No returns becasue output and log files would be saved into './ckpt_dir/' directory.
