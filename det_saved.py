from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import time
import multiprocessing

import numpy as np
from sklearn import datasets, cross_validation, metrics
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

from tensorflow.contrib import skflow

from trainingFunctions import determinant



examples = 20
possibilities = np.zeros(examples)
batchInput = np.zeros(shape=(examples**4,4))
batchTarget = np.zeros(examples**4)

outs = np.zeros(160000)
for i in range(examples):
    possibilities[i] = i

outs_i = 0
target_i = 0
for h in range(examples):  
    for i in range(examples):  
        for j in range(examples):    
            for k in range(examples):
                batchInput[target_i][0] = possibilities[h]
                batchInput[target_i][1] = possibilities[i]
                batchInput[target_i][2] = possibilities[j]
                batchInput[target_i][3] = possibilities[k]
                batchTarget[target_i] = determinant([[batchInput[target_i][0], batchInput[target_i][1]], [batchInput[target_i][2], batchInput[target_i][3]]])
                if(not np.any(outs == batchTarget[target_i])):
                    outs[outs_i] = batchTarget[target_i]
                    outs_i+=1
                target_i += 1
batchTarget = np.rint(batchTarget)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(batchInput, batchTarget,
    test_size=0.1, random_state=42)

regressor = skflow.TensorFlowDNNRegressor(hidden_units=[2000], steps=100000, learning_rate=0.5)

regressor.fit(X_train, y_train)



# score = metrics.mean_squared_error, y_test)

print(y_test)

pred = regressor.predict(X_test)
pred = np.reshape(pred, -1)
pred = np.rint(pred)

print(pred)
error = 1 - accuracy_score(y_test, pred)

print(error)

regressor.save('/home/sanderkd/Data/detSkFlow')


