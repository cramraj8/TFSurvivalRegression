from __future__ import absolute_import, division, print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


'''
Changeable parameters
---------------------
1. learning_rate
2. reg_constant
3. training_epochs
4. batch_size
5. display_step
6. test_size (splitting parameter for train:test)
7. Hidden layer neurons number
    n_hidden_1
    n_hidden_2 = 500
    n_hidden_3 = 500
    n_hidden_4 = 500   
8. variance (for weights & biases, random normal initialization)
9. tf.reduce_sum OR tf.reduce_mean
10. Regularization method
11. Gradient Descend Method
       optimizer = tf.train.GradientDescentOptimizer()  ===> Gives all NaN values
       optimizer = tf.train.AdamOptimizer()             ===> Gives definite values
'''

def read_dataset():

    D = sio.loadmat('Brain_Integ.mat')  
    # Extract the fearues and labels
    X = D['Integ_X'].astype('float32')                                          # 560*399   
    T = np.asarray([t[0] for t in D['Survival']]).astype('float32')             # 560*1
    C = np.asarray([c[0] for c in D['Censored']]).astype('int32')               # 560*1
    # TC = np.asarray([T for c in C if c==1])
    # Consider only the dead patient's data
    TC = T[C==1]            # Only retrieve the data from deceased patients where C == 1
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
# =========== Parameters ============
learning_rate = 0.01                   # Learning rate for parameter updates
reg_constant = 0.01 ; beta = 0.001                    # L2 regularization constant
training_epochs = 1000                  # EPOCHS
batch_size = 100                        # Set to 1 for whole samples at once       ## THIS CAUSES ERROR IN label = batch_y
display_step = 100                      # Set to 1 for displaying all epochs
# dropout_rate = 0.9

# ============ Network Parameters ============
n_hidden_1 = 500                    # 1st layer number of neurons
n_hidden_2 = 500                    # 2nd layer number of neurons
n_hidden_3 = 500                    # 3rd layer number of neurons
n_hidden_4 = 500                    # 4th layer number of neurons
n_input = X_train.shape[1]          # number of columns(features)
n_classes = 1                       # Only one output column, represents the predicted value





# tf Graph input
x = tf.placeholder("float", [None, None], name='features')   # Set to [, None] to definite value
y = tf.placeholder("float", [None], name='labels')         # Set to a definite value

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
# ============================================================= DEFINING REGULARIZATION ========================================================

# # ============================================================= method 1 - wihtout regularizer ===============================================

# # Defining loss and optimizer
# ### reduce_sum ###
# # cost = (tf.reduce_mean(tf.square(tf.transpose(pred)-y)) + 0.01*tf.nn.l2_loss(weights))                        # USE tf.transpose ====> those 2 arrays are row and column, results in unexpected format, if let as early
# cost = tf.reduce_sum(tf.square(tf.transpose(pred)-y), name = "loss")/(2 * total_len)
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Change to tf.GradientDescentOptimizer() and see

# ============================================================= method 2 - using commonly tf.GraphKeys.REGULARIZATION_LOSSES ====================

# loss = tf.reduce_sum(tf.square(tf.transpose(pred)-y), name = "loss")/(2 * total_len)
# # loss = tf.reduce_mean(tf.square(tf.transpose(pred)-y))
# reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)      ## IS THIS L2 REG ???
# cost = loss + reg_constant * sum(reg_losses)
# # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Change to tf.GradientDescentOptimizer() and see
# optimizer  =  tf.train.GradientDescentOptimizer( learning_rate = learning_rate ).minimize (cost)


# ============================================================== method 3 - create regularizer term for each weights==============================


# cost = tf.reduce_mean(tf.square(tf.transpose(pred)-y))
# # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Change to tf.GradientDescentOptimizer() and see
# # Loss function using L2 Regularization
# regularizer = tf.nn.l2_loss(weights['h1'])  # WORKS
# cost = tf.reduce_mean(cost + beta * regularizer)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)


# ============================================================== method 4 - create regularizer term for all weights==============================

loss = tf.reduce_sum(tf.square(tf.transpose(pred)-y), name = "loss")/(2 * total_len)
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Change to tf.GradientDescentOptimizer() and see
# Loss function using L2 Regularization
# regularizer = tf.nn.l2_loss(weights['h1'])
cost = tf.reduce_mean(loss + beta * tf.nn.l2_loss(weights['h1']) + beta * tf.nn.l2_loss(weights['h2']) + beta * tf.nn.l2_loss(weights['h3']) + beta * tf.nn.l2_loss(weights['h4'] + beta * tf.nn.l2_loss(weights['out'])))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# ========================================================== END of Regularization process ========================================================





# Launch the graph
with tf.Session() as sess:
    # Select either Initializers
    # sess.run(tf.initialize_all_variables())
    sess.run(tf.global_variables_initializer())
    # ============================= Each EPOCH graph Session============================================================== 

    for epoch in range(training_epochs+1):
        avg_cost = 0.0
        total_batch = int(total_len/batch_size)
        for i in range(total_batch-1):                                                      # Loop over all batches
            batch_x = X_train[i*batch_size:(i+1)*batch_size]
            batch_y = Y_train[i*batch_size:(i+1)*batch_size]
            # print ("batch number :", i)
            _, c, p = sess.run([optimizer, cost, pred], feed_dict={x: batch_x, y: batch_y}) # Run optimization op (backprop) and cost op (to get loss value)
            avg_cost += c / total_batch                                                     # Compute average loss

        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch : ", "%04d" % (epoch+1), " cost = ", " {:.9f} ".format(avg_cost))
            print ("[*]----------------------------")
            for i in xrange(3):         # Comparing the predicted values with Labeled values in display
                print ("label value:", batch_y[i], "estimated value:", p[i])
            print ("[*]============================")


    # ==================================================================================================================== 
    print ("Optimization Finished!")

    print ("***********************************************")
    print (" \t\t\t TESTING STARTS")
    print ("***********************************************")


    test_cost = sess.run(cost, feed_dict={x:X_test, y: Y_test})
    test_predicted_values = sess.run(pred, feed_dict={x: X_test})
    # print ("Accuracy:", accuracy, "Predicted VALUE : ", predicted_vals)
    for i in range(len(Y_test)):
        print ("Labeled VALUE : ", Y_test[i], " \t\t\t >>>> Predicted VALUE : ", float(test_predicted_values[i]))



    print (" *********************************************** ")
    print (" \t\t\t TESTING ENDS ")
    print (" *********************************************** ")


    # Graphic display
    # plt.plot(X_train, Y_train, 'ro', label='Original data')
    pred_value = sess.run(pred, feed_dict={x: X_train})
    plt.plot(Y_train, pred_value, label='Fitted line')
    plt.legend()
    plt.savefig('plot.png')
    plt.show()
    # plt.close()




