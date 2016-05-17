import tensorflow as tf
import numpy as np 
from trainingFunctions import sine

#Tensorflow session
sess = tf.InteractiveSession()
#Weights and Biases
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
#Placeholder values
x = tf.placeholder(tf.float32, shape=[None, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 1])
#Hidden values
W_hidden = tf.Variable(tf.zeros([1, 10]))
b_hidden = tf.Variable(tf.zeros([1]))
y = tf.nn.softmax(tf.matmul(x,W_hidden) + b_hidden)
#Minimizes the function during training
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#Builtin optimization algorithm
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess.run(tf.initialize_all_variables())

for i in range(10):
	size = 100
	input_ = [None] * size
	target = [None] * size
	for k in range(size):
		input_[k] = np.random.randint(0, 721)
		to_radians = input_[k] * (np.pi/180)
		target[k] = sine(to_radians)
	train_step.run(feed_dict={x: input_, y_: target})