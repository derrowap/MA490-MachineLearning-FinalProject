# Author: Austin Derrow-Pinion
# Purpose: Train a Neural Network to mimic the funcitonality of multiplying
#	two numbers together.
#
# ============================================================================

import tensorflow as tf
import numpy as np
from trainingFunctions import addThem

# ============================================================================
# START TENSORFLOW INTERACTIVE SESSION
# ============================================================================

"""TensorFlow uses a highly efficient C++ backend.

This connection to the backend is called a session.

An interactive session allows you to interleave operations which build a
computation graph with ones that run the graph.

Non-interactive sessions should have the entire computation graph built before
starting a session.
"""
sess = tf.InteractiveSession()

# ============================================================================
# BUILD A FEED-FORWARD NEURAL NETWORK
# ============================================================================

# This builds a feed-forward model with a single hidden layer.

# Placeholders to create nodes for input images and target output classes.
#
# x: 2D tensor of floating point numbers.
#	Dimension 1 is the batch size (any number of example data points).
#	Dimension 2 is an array of two numbers.
# y_: 2D tensor of floating point numbers.
#	Dimension 1 is the batch size (any number of example data points).
#	Dimension 2 is a single value representing the target output.
x = tf.placeholder(tf.float32, shape=[None, 2])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

# Defines weights W and biases b in our model.
#
# Variables are used for model parameters to feed values to.
#
# W: 2x1 matrix of floating points numbers.
# 	Each value represents a connection weight from one input to the output.
# b: Bias variable for the output.
W_hidden = tf.Variable(tf.truncated_normal([2, 2], stddev=0.1))
b_hidden = tf.Variable(tf.truncated_normal([2], stddev=0.1))

# Applies ReLU function to get activation for each hidden node.
h_out = tf.nn.relu(tf.matmul(x, W_hidden) + b_hidden)

keep_prob = tf.placeholder(tf.float32)
h_drop = tf.nn.dropout(h_out, keep_prob)

# Weights and biases from hidden layer to the output layer.
W_out = tf.Variable(tf.truncated_normal([2, 1], stddev=0.1))
b_out = tf.Variable(tf.truncated_normal([1], stddev=0.1))

# Sums activations of output nodes to get final answer.
y = tf.reduce_sum(tf.nn.relu(tf.matmul(h_drop, W_out) + b_out))

# MSE is our cost function to reduce when training.
mse = tf.reduce_mean(tf.square(y - y_))

# Adam optimizer that trains the model.
train_step = tf.train.AdamOptimizer(0.1).minimize(mse)

# Initializes all variables.
#
# Takes the input values (in this case tensors full of zeros) that have been
# specified, and assigns them to each Variable object.
sess.run(tf.initialize_all_variables())

batchSize = 100
batchInput = [None] * batchSize
batchTarget = [None] * batchSize
for i in range(10000):
	for j in range(batchSize):
		a = np.random.randint(1, 10)
		b = np.random.randint(1, 10)
		batchInput[j] = [a, b]
		batchTarget[j] = [addThem(a, b)]
	if i % 1000 == 0:
		print("Iteration %d, MSE = %f" % (i, mse.eval(feed_dict={
			x: batchInput, y_: batchTarget, keep_prob: 1.0})))
	train_step.run(feed_dict={x: batchInput, y_: batchTarget, keep_prob: 0.2})

# Calculates accuracy by using a new set of data.
for j in range(batchSize):
	a = np.random.randint(1, 10)
	b = np.random.randint(1, 10)
	batchInput[j] = [a, b]
	batchTarget[j] = [addThem(a, b)]

print("MSE for multiply function, summing over ReLU:")
print(mse.eval(feed_dict={x: batchInput, y_: batchTarget, keep_prob: 1.0}))

sess.close()