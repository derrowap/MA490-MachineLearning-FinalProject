import numpy as np
import pandas as pd
from tensorflow.contrib.learn import TensorFlowDNNRegressor
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import threading
from trainingFunctions import sine
from sklearn.metrics import mean_squared_error
from tensorflow.contrib import skflow

class myThread(threading.Thread):
	def __init__(self, size):
		threading.Thread.__init__(self)
		self.size = size
		self.x = np.zeros(size)
		self.y = np.zeros(size)
	def run(self):
		for i in range(self.size):
			b = float(np.random.randint(0, 1000))
			a = b * (np.pi/180)
			self.x[i] = a
			self.y[i] = sine(a)
#Create a thread and start it
thread1 = myThread(1000000)
thread1.start()
#Inputs and targets
input_ = thread1.x
target = thread1.y
#Split into training and testing
x_train, x_test, y_train, y_test = train_test_split(input_, target,
	test_size=0.2, random_state=0)
#Neural Network
NN = skflow.TensorFlowDNNRegressor(hidden_units=[2], steps=10000, learning_rate=0.1)
#Fit the training data
NN.fit(x_train, y_train)
#Calculates training error
pred = NN.predict(x_train)
pred = np.reshape(pred, -1)
mse_train = mean_squared_error(y_train, pred)
#Calculates testing error
pred = NN.predict(x_test)
pred = np.reshape(pred, -1)
mse_test = mean_squared_error(y_test, pred)

print("Training mse is: %0.3f" % mse_train)
print("Testing mse is: %0.3f" % mse_test)

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