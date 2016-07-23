from tensorflow.contrib import skflow
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from trainingFunctions import addThem
import numpy as np
import threading

class speciesThread(threading.Thread):
	def __init__(self, ID):
		threading.Thread.__init__(self)
		self.ID = ID
		# self.x = x
		# self.y = y
		self.error = 2
	def run(self):
		self.error = train(self.ID)
		print("Species %d finished with testing error %f"
			% (self.ID, self.error))

# goal = 0.01
# current = 1
# numThreads = 60
# while current > goal:
def trainNet(index):
	# numThreads = high - low + 1

	x = np.zeros((10000 ** 2, 2))
	y = np.zeros(10000 ** 2)

	count = 0
	for i in range(1, 10000):
		for j in range(i, 10000):
			x[count] = [i, j]
			y[count] = addThem(i, j)
			count += 1

	# x_train, x_test, y_train, y_test = train_test_split(x, y,
	# 		test_size=0.2, random_state=0)

	generations = 0
	NN = skflow.TensorFlowEstimator.restore('/home/derrowap/models/addThem'+str(index))
	bestError = 2

	while bestError > 0.001:
		generations += 1
		# pool = [speciesThread(i, x, y) for i in range(low, high+1)]

		# for species in pool:
		# 	species.start()

		# for species in pool:
		# 	species.join()

		# error = [2] * numThreads
		# for i in range(numThreads):
		# 	error[i] = pool[i].error
		# NN = skflow.TensorFlowEstimator.restore('/home/derrowap/models/addThem'+str(ID))
		NN.fit(x, y)

		pred = NN.predict(x)
		pred = np.reshape(pred, -1)
		pred = np.rint(pred)
		error_test = 1 - accuracy_score(y, pred)

		# Update best error so far
		bestError = min(bestError, error_test)

		# bestIndex = np.argmin(error)
		# bestNN = skflow.TensorFlowEstimator.restore('/home/derrowap/models/addThem'+str(pool[bestIndex].ID))
		# for i in range(low, high+1):
		# 	bestNN.save('/home/derrowap/models/addThem'+str(i))
		print("Error on generation %d: %f" % (generations, error_test))
		print("Best error so far: %f" % bestError)
		print("Finished generation %d, continuing...." % generations)
	print("Finished training! Error %f, generations %d." % (bestError, generations))

def train(ID):

	# Neural Network from skflow
	try:
		NN = skflow.TensorFlowEstimator.restore('/home/derrowap/models/addThem'+str(ID))
	except:
		print("ID %d didn't load" % ID)
		NN = skflow.TensorFlowDNNRegressor(hidden_units=[2], steps=100000)

	# Train the NN with training data
	NN.fit(x, y)

	# Calculates training error
	# pred = NN.predict(x_train)
	# pred = np.reshape(pred, -1)
	# pred = np.rint(pred)
	# error_train = 1 - accuracy_score(y_train, pred)

	# Calculates testing error
	pred = NN.predict(x)
	pred = np.reshape(pred, -1)
	pred = np.rint(pred)
	error_test = 1 - accuracy_score(y, pred)

	NN.save('/home/derrowap/models/addThem'+str(ID))
	return(error_test)