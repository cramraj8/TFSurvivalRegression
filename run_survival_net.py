from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import data_providers
import visualize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow.contrib.slim as slim
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging


def run_survival_net(data_file=['./Brain_Integ_X.csv', './Brain_Integ_Y.csv'], ckpt_dir='./ckpt_dir/', split_ratio=0.2, batch_size=100, num_of_epochs_before_decay=1000, n_classes=1, learning_rate_decay_factor=0.7, training_epochs=1000, display_step=100, variance=0.1, BETA=0.001, DROPOUT_RATE=0.9, HIDDEN_LAYERS=[500, 500, 500], INITIAL_LEARNING_RATE=0.0002):

    data_x, data_y = data_providers.data_providers(data_file)
    data_x, data_y = shuffle(data_x, data_y, random_state=1)
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=split_ratio, random_state=420)

    n_observations = x_train.shape[0]             # Total Samples
    input_features = x_train.shape[1]             # number of columns(features)
    N_CLASSES = 1

    with tf.Graph().as_default() as graph:
        logging.set_verbosity(tf.logging.INFO)  # Set the verbosity to INFO level

        if not tf.gfile.Exists(ckpt_dir):
            tf.gfile.MakeDirs(ckpt_dir)  # Needed for files systems in cloud

        num_batches_per_epoch = n_observations / batch_size
        num_steps_per_epoch = num_batches_per_epoch
        decay_steps = int(num_of_epochs_before_decay * num_steps_per_epoch)

        # Create the model
        def multilayer_neural_network_model(inputs, HIDDEN_LAYERS, n_classes, BETA,
                                            scope="deep_regression_model"):
            """Creates a deep regression model.
            Args:
                inputs: A node that yields a `Tensor` of size [total_observations,
                input_features].
            Returns:
                predictions: `Tensor` of shape (1) (scalar) of response.
                end_points: A dict of end points representing the hidden layers.
            """
            with tf.variable_scope(scope, 'deep_regression', [inputs]):
                end_points = {}
                with slim.arg_scope([slim.fully_connected],
                                    activation_fn=tf.nn.relu,
                                    weights_regularizer=slim.l2_regularizer(BETA)):
                    net = slim.stack(inputs,
                                     slim.fully_connected,
                                     HIDDEN_LAYERS,
                                     scope='fc')
                    end_points['fc'] = net
                    predictions = slim.fully_connected(net, n_classes, activation_fn=None,
                                                       scope='prediction')
                    end_points['out'] = predictions
                    return predictions, end_points

        # tf.Graph input
        x = tf.placeholder("float", [None, input_features], name='features')
        y = tf.placeholder("float", [None], name='labels')

        # Construct the grpah for forward pass
        pred, end_points = multilayer_neural_network_model(x, HIDDEN_LAYERS, N_CLASSES, BETA)

        # Create the global step for monitoring the learning_rate and training.
        global_step = get_or_create_global_step()

        # Define your exponentially decaying learning rate
        lr = tf.train.exponential_decay(learning_rate=INITIAL_LEARNING_RATE,
                                        global_step=global_step,
                                        decay_steps=decay_steps,
                                        decay_rate=learning_rate_decay_factor,
                                        staircase=True)

        squared_diff = tf.square(tf.transpose(pred) - y)
        loss = tf.reduce_sum(squared_diff, name="loss") / (2 * n_observations)

        optimize = tf.train.AdamOptimizer(learning_rate=lr)
        optimizer = optimize.minimize(loss, global_step=global_step)

        # Launch the graph
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            # Create a saver object which will save all the variables
            saver = tf.train.Saver()
            for epoch in range(training_epochs + 1):
                avg_cost = 0.0
                n_batches = int(n_observations / batch_size)

                for i in range(n_batches - 1):    # Loop over all batches
                    batch_x = x_train[i * batch_size:(i + 1) * batch_size]
                    batch_y = y_train[i * batch_size:(i + 1) * batch_size]
                    _, batch_cost, batch_pred = sess.run([optimizer,
                                                          loss,
                                                          pred],
                                                         feed_dict={x: batch_x,
                                                                    y: batch_y})
                    # Run optimization (backprop) and cost op (to get loss value)
                    avg_cost += batch_cost / n_batches

                if epoch % display_step == 0:    # Display logs per epoch step
                    print("Epoch : ", "%05d" % (epoch + 1),
                          " cost = ", " {:.9f} ".format(avg_cost))
                    # for i in xrange(3):  # Comparing orig, predicted values
                    #     print("label value:", batch_y[i], "predicted value:",
                    #           batch_pred[i])

            print('Global_Step final value : ', sess.run(global_step))
            # Saving the Graph and Variables in Checkpoint files
            saver.save(sess, ckpt_dir + 'deep_regression_trained_model',
                       global_step=global_step)
            pred_y_train = sess.run(pred, feed_dict={x: x_train})
            print("Training Finished!")

            # TESTING PHASE #
            _, pred_y_test = sess.run([loss, pred], feed_dict={x: x_test, y: y_test})
            for i in range(len(y_test)):
                print("Labeled VALUE : ", y_test[i], " \t\t\t \
                    >>>> Predicted VALUE : ", float(pred_y_test[i]))

    return y_train, y_test, pred_y_train, pred_y_test


if __name__ == '__main__':

    # Pass by flags

    # Contant parameters are defined here
    TRAINING_EPOCHS = 1000                 # EPOCHS
    SPLIT_RATIO = 0.2
    BATCH_SIZE = 100                       # Set to 1 for whole sample at once
    DISPLAY_STEP = 100                     # Set to 1 for displaying all epoch
    VARIANCE = 0.1                         # VARIANCE selection highly affects
    LEARNING_RATE_DECAY_FACTOR = 0.7
    NUM_OF_EPOCHS_BEFORE_DECAY = 1000
    N_CLASSES = 1
    DATA_FILE = ['./Brain_Integ_X.csv', './Brain_Integ_Y.csv']
    CKPT_DIR = './ckpt_dir/'
    # Hyper-parameters
    HYPER_PARAMS = {'BETA': 0.001,
                    'DROPOUT_RATE': 0.9,
                    'HIDDEN_LAYERS': [500, 400, 600, 500, 500, 500],
                    'INITIAL_LEARNING_RATE': 0.0002}

    # Calling the training function
    y_train, y_test, pred_y_train, pred_y_test = run_survival_net(DATA_FILE, CKPT_DIR, SPLIT_RATIO, BATCH_SIZE, NUM_OF_EPOCHS_BEFORE_DECAY, N_CLASSES, LEARNING_RATE_DECAY_FACTOR, TRAINING_EPOCHS, DISPLAY_STEP, VARIANCE, **HYPER_PARAMS)

    # Callling the visualization function
    visualize.visualize(y_train, y_test, pred_y_train, pred_y_test)
