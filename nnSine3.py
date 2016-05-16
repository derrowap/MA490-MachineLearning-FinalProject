import numpy as np
import pandas as pd
from tensorflow.contrib.learn import TensorFlowDNNRegressor
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import threading
from trainingFunctions import sine
from sklearn.metrics import mean_squared_error
from tensorflow.contrib import skflow

#Inputs and targets
input_ = np.zeros(721)
target = np.zeros(721)
for i in range(721):
	b = i
	a = b * (np.pi/180)
	input_[i] = a
	target[i] = sine(a)
#Split into training and testing
#x_train, x_test, y_train, y_test = train_test_split(input_, target,
#	test_size=0.2, random_state=0)
#Neural Network
NN = TensorFlowDNNRegressor(hidden_units=[5,5,5], steps=10000, learning_rate=0.1)
#Fit the training data
NN.fit(input_, target)
#Calculates training error
pred = NN.predict(input_)
#pred = np.reshape(pred, -1)
mse_train = mean_squared_error(target, pred)
#Calculates testing error
#pred = NN.predict(x_test)
#pred = np.reshape(pred, -1)
#mse_test = mean_squared_error(y_test, pred)

print("Training mse is: %0.3f" % mse_train)
#print("Testing mse is: %0.3f" % mse_test)

# Enters loop to predict inputs from user.
print("\nEnter exit to leave loop.")
while True:
	input_ = input("Enter degrees: ")
	try:
		# succeeds if user typed a number
		input_ = int(input_)
		input_radians = input_ * (np.pi/180)
	except:
		# exit loop
		break
	# Calculates prediction from NN
	result = NN.predict(np.array([input_radians]))
	print("I think the sine of %d degrees = %0.3f, the actual value is %0.3f."
		% (input_, result[0], sine(input_radians)))