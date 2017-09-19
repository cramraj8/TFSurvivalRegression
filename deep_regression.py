from __future__ import absolute_import, division, print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow.contrib.slim as slim
from tensorflow.contrib.framework.python.ops.variables \
    import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging


'''
Changeable parameters OR variables
----------------------------------
1. LEARNING_RATE
2. reg_constant
3. TRAINING_EPOCHS
4. BATCH_SIZE
5. DISPLAY_STEP
6. test_size (splitting parameter for train:test)
7. Hidden layer neurons number
    N_HIDDEN_1
    N_HIDDEN_2
    N_HIDDEN_3
    N_HIDDEN_4
8. VARIANCE (for weights & biases, random normal initialization)
9. tf.reduce_sum OR tf.reduce_mean
10. Regularization method
11. Gradient Descend Method
    optimizer = tf.train.GradientDescentOptimizer()  ===> Gives all NaN values
    optimizer = tf.train.AdamOptimizer()             ===> Gives definite values
'''


# =========== Parameters ============
# LEARNING_RATE = 0.01                      # Learning rate
BETA = 0.001                                # L2 regularization constant
TRAINING_EPOCHS = 1000                       # EPOCHS
BATCH_SIZE = 100                            # Set to 1 for whole sample at once
DISPLAY_STEP = 100                          # Set to 1 for displaying all epoch
DROPOUT_RATE = 0.9
VARIANCE = 0.1                              # VARIANCE selection highly affects
# Learning rate information and configuration (Up to you to experiment)
INITIAL_LEARNING_RATE = 0.0002
LEARNING_RATE_DECAY_FACTOR = 0.7
NUM_OF_EPOCHS_BEFORE_DECAY = 1000

# ============ Network Parameters ============
HIDDEN_LAYERS = [500, 400, 600, 500]
N_CLASSES = 1                       # Only one output prediction column


def load_data_set(name=None):
    """
    This function reads the data file and extracts the features and labelled
    values.
    Then according to that patient is dead, only those observations would be
    taken care
    for deep learning trainig.
  Args:
    Nothing
  Returns:
    `Numpy array`, extracted feature columns and label column.
  Example:
    >>> read_dataset()
    ( [[2.3, 2.4, 6.5],[2.3, 5.4,3.3]], [12, 82] )
    """

    data = sio.loadmat('Brain_Integ.mat')  # Extract the fearues and labels
    fearues = data['Integ_X'].astype('float32')                       # 560*399
    time_of_death = np.asarray([t[0] for t in data['Survival']])\
        .astype('float32')  # 560*1
    censored = np.asarray([c[0] for c in data['Censored']])\
        .astype('int32')    # 560*1
    censored_time_of_death = time_of_death[censored == 1]
    # Only retrieve the data from deceased patients where C == 1
    censored_features = fearues[censored == 1]
    y = np.asarray(censored_time_of_death)
    x = np.asarray(censored_features)

    print('Shape of X : ', x.shape)
    print('Shape of Y : ', y.shape)
    return (x, y)


# ******************************************************************************


with tf.Graph().as_default() as graph:
    logging.set_verbosity(tf.logging.INFO)  # Set the verbosity to INFO level

    data_x, data_y = load_data_set()
    data_x, data_y = shuffle(data_x, data_y, random_state=1)
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y,
                                                        test_size=0.20,
                                                        random_state=420)

    total_observations = x_train.shape[0]         # Total Samples
    input_features = x_train.shape[1]             # number of columns(features)

    log_dir = './log/'
    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)  # Needed for files systems in cloud

    ckpt_dir = './regression_model/'
    if not tf.gfile.Exists(ckpt_dir):
        tf.gfile.MakeDirs(ckpt_dir)

    num_batches_per_epoch = total_observations / BATCH_SIZE
    num_steps_per_epoch = num_batches_per_epoch
    decay_steps = int(NUM_OF_EPOCHS_BEFORE_DECAY * num_steps_per_epoch)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Create the model inference
    def multilayer_neural_network_model(inputs, HIDDEN_LAYERS, BETA,
                                        scope="deep_regression_model"):
        """Creates a deep regression model.

        Args:
            inputs: A node that yields a `Tensor` of size [total_observations,
            input_features].

        Returns:
            predictions: `Tensor` of shape [1] (scalar) of response.
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
                predictions = slim.fully_connected(net, 1, activation_fn=None,
                                                   scope='prediction')
                end_points['out'] = predictions
                return predictions, end_points

    # tf.Graph input
    x = tf.placeholder("float", [None, input_features], name='features')
    y = tf.placeholder("float", [None], name='labels')

    # Construct the grpah for forward pass
    pred, end_points = multilayer_neural_network_model(x, HIDDEN_LAYERS, BETA)

    # Print name and shape of each tensor.
    print("Layers")
    for k, v in end_points.items():
        print('name = {}, shape = {}'.format(v.name, v.get_shape()))

    # Print name and shape of parameter nodes  (values not yet initialized)
    print("Parameters")
    for v in slim.get_model_variables():
        print('name = {}, shape = {}'.format(v.name, v.get_shape()))

    # Create the global step for monitoring the learning_rate and training.
    global_step = get_or_create_global_step()

    # Define your exponentially decaying learning rate
    lr = tf.train.exponential_decay(learning_rate=INITIAL_LEARNING_RATE,
                                    global_step=global_step,
                                    decay_steps=decay_steps,
                                    decay_rate=LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

# ********************** LOSS && OPTIMIZE *************************************

    mean_squared_error_loss = tf.reduce_sum(
        tf.square(tf.transpose(pred) - y),
        name="loss") / (2 * total_observations)

    regularization_loss = tf.add_n(tf.losses.get_regularization_losses())

    total_loss = mean_squared_error_loss + regularization_loss
    # total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

#                   ************************************

    optimizer = tf.train.AdamOptimizer(learning_rate=lr).\
        minimize(total_loss, global_step=global_step)

# ******************************************************************************
    # Launch the graph
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        # Create a saver object which will save all the variables
        saver = tf.train.Saver()

        for epoch in range(TRAINING_EPOCHS + 1):
            avg_cost = 0.0
            total_batch = int(total_observations / BATCH_SIZE)
            for i in range(total_batch - 1):    # Loop over all batches
                batch_x = x_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                batch_y = y_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                _, batch_cost, batch_pred = sess.run([optimizer,
                                                      total_loss,
                                                      pred],
                                                     feed_dict={x: batch_x,
                                                                y: batch_y})
                # Run optimization (backprop) and cost op (to get loss value)
                avg_cost += batch_cost / total_batch   # Compute average loss

            if epoch % DISPLAY_STEP == 0:         # Display logs per epoch step
                print("Epoch : ", "%05d" % (epoch + 1),
                      " cost = ", " {:.9f} ".format(avg_cost))
                for i in xrange(3):  # Comparing orig, predicted values
                    print("label value:", batch_y[i], "predicted value:",
                          batch_pred[i])
                print("------------------------------------------------------")
        print('Global_Step final value : ', sess.run(global_step))
        # Saving the Graph and Variables in Checkpoint files
        saver.save(sess, ckpt_dir + 'deep_regression_trained_model',
                   global_step=global_step)
        pred_out = sess.run(pred, feed_dict={x: x_train})

        print("Training Finished!")

        # TESTING PHASE #
        _, test_predicted_values = sess.run([total_loss, pred],
                                            feed_dict={x: x_test,
                                                       y: y_test})
        for i in range(len(y_test)):
            print("Labeled VALUE : ", y_test[i], " \t\t\t \
                >>>> Predicted VALUE : ", float(test_predicted_values[i]))


#### ------------------ PLOTTING -------------------------------------- ####
plt.plot(y_train, pred_out, 'ro',
         label='Correlation of Original with Predicted train data')
# Above is for marking points in space of labels and features
plt.plot(y_test, test_predicted_values, 'bo',
         label='Correlation of Original with Predicted test data')
plt.title('Comparison of the predicting performance')
plt.ylabel('Predicted data')
plt.xlabel('Original data')
plt.savefig('Correlation.png')
plt.legend()    # This enable the label over the plot display
plt.show()      # This is important for showing the plotted graph
