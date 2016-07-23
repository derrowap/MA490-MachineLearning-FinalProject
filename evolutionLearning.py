# Author: Austin Derrow-Pinion
# Purpose: Create an evolutionary algorithm to teach nueral networks.
#
# ============================================================================

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import threading

class speciesThread(threading.Thread):
	def __init__(self, NN, ID, sizeData, function, low, high):
		threading.Thread.__init__(self)
		self.NN = NN
		self.ID = ID
		self.sizeData = sizeData
		self.function = function
		self.low = low
		self.high = high
		self.error = 0
	def run(self):
		# Generates the data from function
		x = np.zeros((self.sizeData, 2))
		y = np.zeros(self.sizeData)
		for i in range(self.sizeData):
			a = float(np.random.randint(self.low, self.high))
			b = float(np.random.randint(self.low, self.high))
			x[i] = [a, b]
			y[i] = self.function(a, b)

		# Splits generated data into training and testing
		x_train, x_test, y_train, y_test = train_test_split(x, y, 
			test_size=0.2, random_state=0)

		# Teaches the NN with data
		self.NN.fit(x_train, y_train)

		# Calculates testing error
		pred = self.NN.predict(x_test)
		pred = np.reshape(pred, -1)
		pred = np.rint(pred)
		self.error = 1 - accuracy_score(y_test, pred)
		print("Species %d finished training with test error %f"
			% (self.ID, self.error))

def evolve(NN, numSpecies, numGenerations, func, sizeData, low, high):
	winner = NN
	lowestError = 2
	for i in range(numGenerations):
		# Creates an array of species, each threaded
		pool = [None] * numSpecies
		for j in range(numSpecies):
			pool[j] = speciesThread(winner, j, sizeData, func, low, high)

		# Starts training each species
		# for species in pool:
		# 	species.start()
		for j in range(numSpecies):
			pool[j].start()

		# Waits for each species to be finished training
		# for species in pool:
		# 	species.join()
		for j in range(numSpecies):
			pool[j].join()

		# Re-assign the winner of the evolutionary process to the species
		# with the lowest testing error.
		winningID = -1
		# for species in pool:
		# 	if species.error < lowestError:
		# 		winner = species.NN
		# 		lowestError = species.error
		# 		winningIndex = count
		# 	count += 1
		for j in range(numSpecies):
			if pool[j].error < lowestError:
				winner = pool[j].NN
				lowestError = pool[j].error
				winningID = pool[j].ID
		if winningID != -1:
			print("Winner: Generation %d / %d, Species %d / %d,\
				testing error %f"
				% (i, numGenerations, winningID, numSpecies, lowestError))
		else:
			print("Generation %d / %d had no new winners. Continuing with\
				current winning species with test error %f"
				% (i, numGenerations, lowestError))
	print("Finished evolving! After %d generations, the winning species has\
		a testing error of %f" % (numGenerations, lowestError))
	return(winner)