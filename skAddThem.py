# Author: Austin Derrow-Pinion
# Purpose: Train a Neural Network to mimic the funcitonality of adding
#	two numbers together.
#
# ============================================================================

import numpy as np
from tensorflow.contrib.learn import TensorFlowDNNRegressor
from trainingFunctions import addThem
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

size = 1000
x = np.zeros((size ** 2, 2))
y = np.zeros(size ** 2)
count = 0
for i in range(size):
	for j in range(size):
		x[count] = [i, j]
		y[count] = addThem(i, j)
		count += 1

x_train, x_test, y_train, y_test = train_test_split(
	x, y, test_size=0.1, random_state=0)


numSteps = 100000
NN = TensorFlowDNNRegressor(hidden_units=[2], steps=numSteps)

NN.fit(x_train, y_train)
pred = NN.predict(x_test)
pred = np.reshape(pred, -1)
pred = np.rint(pred)
error = 1 - accuracy_score(y_test, pred)
print('Steps %d, error %f' % (numSteps, error))

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
		% (first, second, int(result)))
	print("True answer: %d + %d = %d"
		% (first, second, addThem(first, second)))