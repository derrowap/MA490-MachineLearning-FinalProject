# Author: Austin Derrow-Pinion
# Purpose: Train a Neural Network to mimic the funcitonality of adding
#	two numbers together.
#
# Results:
# Numbers between 1 and 100 gives perfect accuracy.
#
# Numbers between 1 and 500 gives 0.42 error.
#
# Numbers between 1 and 1000 gives 0.87 error.
#
# With numbers between 1 and 500, the output is off by 1 whenever the sum is
# greater than 620.
# 
# The best architecture for the neural network is a single hidden layer of 2
# nodes.
# ============================================================================

import numpy as np
import pandas as pd
from tensorflow.contrib import skflow
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
# import threading
from trainingFunctions import addThem

# class myThread(threading.Thread):
# 	def __init__(self, size):
# 		threading.Thread.__init__(self)
# 		self.size = size
# 		self.x = np.zeros((size, 2))
# 		self.y = np.zeros(size)
# 	def run(self):
# 		for i in range(self.size):
# 			a = float(np.random.randint(1, 500))
# 			b = float(np.random.randint(1, 500))
# 			self.x[i] = [a, b]
# 			self.y[i] = addThem(a, b)

def train(ID):

	# # Create new threads
	# thread1 = myThread(1000000)
	# thread2 = myThread(1000000)

	# # Start new Threads
	# thread1.start()
	# thread2.start()

	# # Wait for threads to complete
	# thread1.join()
	# thread2.join()

	# # Combine two threads output together
	# input_ = thread1.x + thread2.x
	# target = thread1.y + thread2.y

	size = 1000000
	x = np.zeros((size, 2))
	y = np.zeros(size)
	for i in range(size):
		a = float(np.random.randint(1, 500))
		b = float(np.random.randint(1, 500))
		x[i] = [a, b]
		y[i] = addThem(a, b)

	x_train, x_test, y_train, y_test = train_test_split(x, y,
		test_size=0.2, random_state=0)

	# Neural Network from skflow
	try:
		NN = skflow.TensorFlowEstimator.restore('/home/derrowap/models/addThem'+str(ID))
	except:
		NN = skflow.TensorFlowDNNRegressor(hidden_units=[2], steps=5000)

	# Train the NN with training data
	NN.fit(x_train, y_train)

	# Calculates training error
	# pred = NN.predict(x_train)
	# pred = np.reshape(pred, -1)
	# pred = np.rint(pred)
	# error_train = 1 - accuracy_score(y_train, pred)

	# Calculates testing error
	pred = NN.predict(x_test)
	pred = np.reshape(pred, -1)
	pred = np.rint(pred)
	error_test = 1 - accuracy_score(y_test, pred)

	NN.save('/home/derrowap/models/addThem'+str(ID))
	return(error_test)

# NN = train(-1)

NN = skflow.TensorFlowEstimator.restore('/home/derrowap/models/addThem1')
# Enters loop to predict inputs from user.
print("\nEnter exit to leave loop.")
while True:
	first = input("Number 1... ")
	try:
		# succeeds if user typed a number
		first = int(first)
	except:
		# exit loop
		break
	second = input("Number 2... ")
	try:
		# succeeds if user typed a number
		second = int(second)
	except:
		# exit loop
		break
	# Calculates prediction from NN
	result = NN.predict(np.array([[first, second]]))
	print("I think %d + %d = %d"
		% (first, second, int(np.rint(result[0][0]))))