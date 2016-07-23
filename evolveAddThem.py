# Author: Austin Derrow-Pinion
# Purpose: Train a Neural Network to mimic the funcitonality of adding
#	two numbers together.
#
# ============================================================================

import numpy as np
from tensorflow.contrib.learn import TensorFlowDNNRegressor
import evolutionLearning as e
from trainingFunctions import addThem

# Initializes the Neural Network to evolve.
NN = TensorFlowDNNRegressor(hidden_units=[2], steps=5000)

# Evolves the neural network to learn the function.
# Params:
# 	NN: NN - neural network to evolve.
# 	numSpecies: 10 - number of species to train at a time.
# 	numGenerations: 10 - number of generations to evolve through.
# 	func: addThem - function to train the neural network on.
# 	sizeData: 1000000 - number of data examples each species will generate
#		and train on.
# 	low: 1 - smallest number to generate in the data.
# 	high: 500 - largest number to generate in the data.
NN = e.evolve(NN, 20, 2, addThem, 100000, 1, 1000)

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