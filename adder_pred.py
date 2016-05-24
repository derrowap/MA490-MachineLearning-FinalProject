from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import time

import numpy as np
from sklearn import datasets, cross_validation, metrics
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

from tensorflow.contrib import skflow

from trainingFunctions import adder


batchSize = 1000
batchInput = np.zeros(batchSize)
batchTarget = np.zeros(batchSize)
i = 0
for j in range(batchSize):
    i+=1
    batchInput[j] = i
    batchTarget[j] = adder(batchInput[j])
    if(i == 100):
        i = 0
print(batchInput)
print(batchTarget)


testSize = 1000
testInput = np.zeros(testSize)
testTarget = np.zeros(testSize)
for i in range(testSize):
    testInput[i] = i*1000
    testTarget[i] = adder(testInput[i])


reg = skflow.TensorFlowDNNRegressor(hidden_units=[2], steps=100, learning_rate=0.1)

reg.fit(batchInput, batchTarget)

pred = reg.predict(testInput)

arr = np.array(pred)

score = metrics.mean_squared_error(reg.predict(batchInput), batchTarget)
print('%d MSE: %f' % (i, score))

np.savetxt('/home/sanderkd/Data/adder_pred_2.csv', (testInput, np.rint(arr[:,0])), delimiter=', ')
reg.save('/home/sanderkd/Data/adder_pred')

while True:
    val = int(input("Enter val to add: "))
    prediction = reg.predict(np.array([[val]]))
    print("prediction: %d" % round(prediction[0][0]))