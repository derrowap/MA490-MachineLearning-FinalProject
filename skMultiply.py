# Author: Austin Derrow-Pinion
# Purpose: Train a Neural Network to mimic the funcitonality of multiplying
#	two numbers together.
#
# ============================================================================

import numpy as np
from tensorflow.contrib.learn import TensorFlowDNNRegressor
from trainingFunctions import multiply
from sklearn.metrics import accuracy_score

top = 20
x = np.zeros((top ** 2, 2))
y = np.zeros(top ** 2)
count = 0
for i in range(1, top+1):
	for j in range(1, top+1):
		x[count] = [i, j]
		y[count] = multiply(i, j)
		count += 1

numSteps = 500000
NN = TensorFlowDNNRegressor(hidden_units=[400], steps=numSteps)

NN.fit(x, y)
pred = NN.predict(x)
pred = np.reshape(pred, -1)
pred = np.rint(pred)
error = 1 - accuracy_score(y, pred)
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
	print("I think %d * %d = %d"
		% (first, second, int(np.rint(result[0][0]))))