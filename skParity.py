# Author: Austin Derrow-Pinion
# Purpose: Train a Neural Network to mimic the funcitonality of multiplying
#	two numbers together.
#
# ============================================================================

import numpy as np
from tensorflow.contrib.learn import TensorFlowDNNClassifier
from trainingFunctions import evenParity
from sklearn.metrics import accuracy_score

x = np.zeros((65536, 16), dtype='int32')
y = np.zeros(65536, dtype='int32')
for i in range(65536):
	temp = bin(i)[2:].zfill(16)
	x[i] = [int(j) for j in temp]
	y[i] = evenParity(i)

numSteps = 1000000
NN = TensorFlowDNNClassifier(hidden_units=[16], steps=numSteps,
	n_classes=2)

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
	result = NN.predict(np.array([[int(j) for j in bin(first)[2:].zfill(16)]]))
	print("I think evenParity(%d) = %d"
		% (first, int(result)))
	print("True answer of evenParity(%d) = %d" % (first, evenParity(first)))