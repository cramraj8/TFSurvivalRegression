from __future__ import absolute_import, division, print_function

import tensorflow as tf

# import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def read_dataset():

    D = sio.loadmat('Brain_Integ.mat')  
    # Extract the fearues and labels
    X = D['Integ_X'].astype('float32')                                          # 560*399   
    T = np.asarray([t[0] for t in D['Survival']]).astype('float32')             # 560*1
    C = np.asarray([c[0] for c in D['Censored']]).astype('int32')       # O # 560*1
    # TC = np.asarray([T for c in C if c==1])
    # Consider only the dead patient's data
    TC = T[C==1]
    XC = X[C==1]
    Y = np.asarray(TC)
    X = np.asarray(XC)

    print ('Shape of X : ', X.shape)
    print ('Shape of Y : ', Y.shape)
    return (X, Y)

X, Y = read_dataset()
X, Y = shuffle(X, Y, random_state=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=420)


total_len = X_train.shape[0]        # Total Samples

# ***** Parameters *****
learning_rate = 0.01               # Learning rate for parameter updates
training_epochs = 1000               # EPOCHS
batch_size = 100                      # Set to 1 for whole samples at once       ## THIS CAUSES ERROR IN label = batch_y
display_step = 100                    # Set to 1 for displaying all epochs
# dropout_rate = 0.9

# ***** Network Parameters *****
n_hidden_1 = 500                    # 1st layer number of neurons
n_hidden_2 = 500                    # 2nd layer number of neurons
n_hidden_3 = 500                    # 3rd layer number of neurons
n_hidden_4 = 500                    # 4th layer number of neurons
n_input = X_train.shape[1]          # number of columns(features)
n_classes = 1                       # Only one output column, represents the predicted value

# tf Graph input
x = tf.placeholder("float", [None, None])   # Set to [, None] to definite value
y = tf.placeholder("float", [None])         # Set to a definite value

# layers' weights & biases initialization
variance = 0.1      # VARIANCE selection highly affects     ## If set to 10, results in NaN values for predication and cost
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], 0, variance)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], 0, variance)),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], 0, variance)),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], 0, variance)),
    'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes], 0, variance))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], 0, variance)),
    'b2': tf.Variable(tf.random_normal([n_hidden_2], 0, variance)),
    'b3': tf.Variable(tf.random_normal([n_hidden_3], 0, variance)),
    'b4': tf.Variable(tf.random_normal([n_hidden_4], 0, variance)),
    'out': tf.Variable(tf.random_normal([n_classes], 0, variance))
}

# print (weights['h1'])
# print (biases['b1'])
# Model Creation
def multilayer_perceptron(x, weights, biases):
    # Hidden layers with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)

    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer


# Construct the grpah for forward pass
pred = multilayer_perceptron(x, weights, biases)

# Defining loss and optimizer
### reduce_sum ###
# cost = (tf.reduce_mean(tf.square(tf.transpose(pred)-y)) + 0.01*tf.nn.l2_loss(weights))                        # USE tf.transpose ====> those 2 arrays are row and column, results in unexpected format, if let as early
cost = tf.reduce_mean(tf.square(tf.transpose(pred)-y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Change to tf.GradientDescentOptimizer() and see




# Launch the graph
with tf.Session() as sess:

    # Select either Initializers
    sess.run(tf.initialize_all_variables())
    # tf.global_variables_initializer()
    # ============================= Each EPOCH graph Session============================================================== 

    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = int(total_len/batch_size)
        # Loop over all batches
        for i in range(total_batch-1):
            batch_x = X_train[i*batch_size:(i+1)*batch_size]
            batch_y = Y_train[i*batch_size:(i+1)*batch_size]
            print ("batch number :", i)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, p = sess.run([optimizer, cost, pred], feed_dict={x: batch_x, y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch


        # sample prediction
        label_value = batch_y
        estimate = p    # Predicted value
        err = label_value-estimate
        # print ("num batch:", total_batch)

        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", '%4d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            print ("[*]----------------------------")
            for i in xrange(3):         # Comparing the predicted values with Labeled values in display
                print ("label value:", label_value[i], "estimated value:", estimate[i])
            print ("[*]============================")

    # ==================================================================================================================== 
    print ("Optimization Finished!")

    # # Test model
    # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # # Calculate accuracy
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # print ("Accuracy:", accuracy.eval({x: X_test, y: Y_test}))

    print ("TESTING STARTS")

    accuracy = sess.run(cost, feed_dict={x:X_test, y: Y_test})
    predicted_vals = sess.run(pred, feed_dict={x: X_test})
    # print ("Accuracy:", accuracy, "Predicted VALUE : ", predicted_vals)
    print ("Labeled value : ", Y_test, " <<<>>> Predicted VALUE : ", predicted_vals)

    print ("TESTING ENDS")



