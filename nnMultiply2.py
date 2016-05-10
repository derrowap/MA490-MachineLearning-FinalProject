# Author: Austin Derrow-Pinion
# Purpose: Train a Neural Network to mimic the funcitonality of multiplying
#	two numbers together.
#
# ============================================================================

import tensorflow as tf
import numpy as np
from trainingFunctions import multiply

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

def weight_variable(shape):
	"""Defines a Variable with specified shape as weights.

	Initializes with a little noise to prevent symmetry breaking and 0
	gradients.

	Args:
		shape: Shape of the weight Variable to define.

	Returns:
		Variable of specified shape containing weights initialized with noise.
	"""
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	"""Defines a Variabel with specified shape as biases.

	All bias variables are initialized to 0.1.

	Args:
		shape: Shape of the bias Variable to define.

	Returns:
		Variable of specified shape containing biases initialized to 0.1.
	"""
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


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
W_hidden = tf.Variable(tf.zeros([2, 100]))
b_hidden = tf.Variable(tf.zeros([100]))

# Applies ReLU function to get activation for each hidden node.
h_out = tf.nn.relu(tf.matmul(x, W_hidden) + b_hidden)

# Sums activations of hidden node to get final output.
y = tf.reduce_sum(h_out)


# Defines cross entropy cost function to minimize.
#
# tf.reduce_sum sums across all classes.
# tf.reduce_mean takes the average over these sums.
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
#	reduction_indices=[1]))

mse = tf.reduce_mean(tf.square(y - y_))


train_step = tf.train.AdamOptimizer(1e-4).minimize(mse)

# Initializes all variables.
#
# Takes the input values (in this case tensors full of zeros) that have been
# specified, and assigns them to each Variable object.
sess.run(tf.initialize_all_variables())

for i in range(10000):
	batchSize = 10
	batchInput = [None] * batchSize
	batchTarget = [None] * batchSize
	for j in range(batchSize):
		batchInput[j] = [np.random.randint(1, 10), np.random.randint(1, 10)]
		batchTarget[j] = [multiply(batchInput[j][0], batchInput[j][1])]

# Calculates accuracy by using a new set of data.
print("MSE for multiply function, summing over ReLU:")
print(mse.eval(feed_dict={x: batchInput, y_: batchTarget}))

sess.close()