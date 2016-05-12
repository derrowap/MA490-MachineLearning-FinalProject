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
from tensorflow.contrib.learn import TensorFlowDNNRegressor
from trainingFunctions import addThem
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

size = 2000000
input_ = np.zeros((size, 2))
target = np.zeros(size)
for i in range(size):
	a = float(np.random.randint(1, 500))
	b = float(np.random.randint(1, 500))
	input_[i] = [a, b]
	target[i] = addThem(a, b)

x_train, x_test, y_train, y_test = train_test_split(input_, target,
	test_size=0.2, random_state=0)

NN = TensorFlowDNNRegressor(hidden_units=[2], steps=10000,
	learning_rate=0.1)

NN.fit(x_train, y_train)
pred = NN.predict(x_train)
pred = np.reshape(pred, -1)
pred = np.rint(pred)
error_train = 1 - accuracy_score(y_train, pred)

pred = NN.predict(x_test)
pred = np.reshape(pred, -1)
pred = np.rint(pred)
error_test = 1 - accuracy_score(y_test, pred)

print('\nTraining error = %0.3f    testing error = %0.3f'
	% (error_train, error_test))

print("\nEnter exit to leave loop.")
while True:
	first = input("Number 1... ")
	try:
		first = int(first)
	except:
		break
	second = input("Number 2... ")
	try:
		second = int(second)
	except:
		break
	result = NN.predict(np.array([[first, second]]))
	print("I think %d + %d = %d"
		% (first, second, int(np.rint(result[0][0]))))