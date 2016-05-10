# Author: Austin Derrow-Pinion
# Purpose: Train a Neural Network to mimic the funcitonality of multiplying
#	two numbers together.
#
# Result: This is a softmax neural network that computes a probability
# 	distribution over outputs 1 to 100. It is a single linear layer from input
# 	to the output. Multiplying two numbers together is not linear, therefore
# 	as expected, the nueral network did very poorly at learning this function.
# 	Adding a hidden layer is necessary in order to do better. Also, using
# 	the softmax function doesn't seem like the right idea since this is not a
# 	classification problem, but a regression.
# 	Estimated testing accuracy = 16%.
# ============================================================================

import tensorflow as tf
import numpy as np
from trainingFunctions import multiply

def one_hot(index):
	"""Defines a one-hot encoding array for specified index.

	Args:
		index: The index in the array to assign as a 1.

	Returns:
		An array filled with zeros, except a 1 at the specified index.
	"""
	output = np.zeros(100)
	output[index] = 1
	return output

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
# BUILD A SOFTMAX REGRESSION MODEL
# ============================================================================

# This builds a softmax regression model with a single linear layer.

# Placeholders to create nodes for input images and target output classes.
#
# x: 2D tensor of floating point numbers.
#	Dimension 1 is the batch size (any number of example data points).
#	Dimension 2 is an array of two numbers.
# y_: 2D tensor of floating point numbers.
#	Dimension 1 is the batch size (any number of example data points).
#	Dimension 2 is a single value representing the target output.
x = tf.placeholder(tf.float32, shape=[None, 2])
y_ = tf.placeholder(tf.float32, shape=[None, 100])

# Defines weights W and biases b in our model.
#
# Variables are used for model parameters to feed values to.
#
# W: 2x1 matrix of floating points numbers.
# 	Each value represents a connection weight from one input to the output.
# b: Bias variable for the output.
W = tf.Variable(tf.zeros([2, 100]))
b = tf.Variable(tf.zeros([100]))

# Initializes all variables.
#
# Takes the input values (in this case tensors full of zeros) that have been
# specified, and assigns them to each Variable object.
sess.run(tf.initialize_all_variables())

# Defines softmax regression model.
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Defines cross entropy cost function to minimize.
#
# tf.reduce_sum sums across all classes.
# tf.reduce_mean takes the average over these sums.
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
	reduction_indices=[1]))

# ============================================================================
# TRAIN THE MODEL
# ============================================================================

# Defines steepest gradient descent optimization algorithm.
#
# Step length of 0.5 and minimizes cross_entropy cost function.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Trains gradient descent 1000 times using 50 training examples each loop.
for i in range(10000):
	batchSize = 10
	batchInput = [None] * batchSize
	batchTarget = [None] * batchSize
	for j in range(batchSize):
		batchInput[j] = [np.random.randint(1, 10), np.random.randint(1, 10)]
		batchTarget[j] = one_hot(multiply(batchInput[j][0], batchInput[j][1]))
	train_step.run(feed_dict={x: batchInput, y_: batchTarget})

# ============================================================================
# TEST THE MODEL
# ============================================================================

# Create boolean array indicating if model predicts them correctly.
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# Defines accuracy as a percent by converting booleans to 0's and 1's.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Calculates accuracy by using a new set of examples.
batchSize = 50
batchInput = [None] * batchSize
batchTarget = [None] * batchSize
for j in range(batchSize):
	batchInput[j] = [np.random.randint(1, 10), np.random.randint(1, 10)]
	batchTarget[j] = one_hot(multiply(batchInput[j][0], batchInput[j][1]))

# Calculates accuracy by using a new set of data.
print("Accuracy from softmax regression model with a single linear layer:")
print(accuracy.eval(feed_dict={x: batchInput, y_: batchTarget}))

# close interactive session
sess.close()