import numpy as np
import pandas as pd
from tensorflow.contrib.learn import TensorFlowDNNRegressor
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import threading
from trainingFunctions import sine
from trainingFunctions import expansionTerms
from sklearn.metrics import mean_squared_error
from tensorflow.contrib import skflow

#Inputs and targets
range_ = np.linspace(-3 * np.pi, 3 * np.pi, num=10000, dtype = 'float32')
input_ = np.zeros((10000,9), dtype='float32')
target = np.zeros(10000, dtype='float32')
for i in range(10000):
	input_[i] = expansionTerms(range_[i])
	target[i] = sine(range_[i])
#Split into training and testing
#x_train, x_test, y_train, y_test = train_test_split(input_, target,
#	test_size=0.2, random_state=0)
#Neural Network
NN = TensorFlowDNNRegressor(hidden_units=[9], steps=100000)
#Fit the training data
NN.fit(input_, target)
pred = NN.predict(input_)
#pred = np.reshape(pred, -1)
mse_train = mean_squared_error(target, pred)
#Calculates testing error
#pred = NN.predict(x_test)
#pred = np.reshape(pred, -1)
#mse_test = mean_squared_error(y_test, pred)
range_2 = np.linspace(-5*np.pi, 5*np.pi, num = 100, dtype = 'float32')
input_2 = np.zeros((100,9), dtype='float32')
for i in range(100):
	input_2[i] = expansionTerms(range_2[i])

pred2 = NN.predict(input_2)

np.savetxt('/home/barteeaj/Data/Input.csv',range_2)
np.savetxt('/home/barteeaj/Data/Pred.csv',pred2)

print("Training mse is: %0.3f" % mse_train)
#print("Testing mse is: %0.3f" % mse_tes
#Calculates training errort)

# Enters loop to predict inputs from user.
print("\nEnter exit to leave loop.")
while True:
	input_ = input("Enter radians: ")
	try:
		# succeeds if user typed a number
		input_ = float(input_)
	except:
		# exit loop
		break
	# Calculates prediction from NN
	result = NN.predict(np.array([expansionTerms(input_)]))
	print("I think the sine of %0.3f = %0.3f, the actual value is %0.3f."
		% (input_, result[0][0], sine(input_)))