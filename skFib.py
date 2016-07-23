# Author: Austin Derrow-Pinion
# Purpose: Train a Neural Network to mimic the funcitonality of multiplying
#	two numbers together.
#
# ============================================================================

import numpy as np
from tensorflow.contrib.learn import TensorFlowDNNRegressor
from trainingFunctions import fib
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

x = np.zeros(70)
y = np.zeros(70, dtype='int64')
for i in range(70):
	x[i] = i + 1
	y[i] = fib(i + 1)

x = normalize(x, norm='max')

numSteps = 10000
NN = TensorFlowDNNRegressor(hidden_units=[100, 100, 100], steps=numSteps)

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
		first = int(first) / 70
	except:
		break
	result = NN.predict(np.array([first]))
	print("I think fib_%d = %d"
		% (first, int(np.rint(result[0][0]))))