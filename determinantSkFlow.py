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

from trainingFunctions import determinant



examples = 1000
batchInput = np.zeros(shape=(examples,4))
batchTarget = np.zeros(examples)
for i in range(examples):
    one = np.random.random_sample()
    two = np.random.random_sample()
    three = np.random.random_sample()
    four = np.random.random_sample()
    batchInput[i][0] = one
    batchInput[i][1] = two
    batchInput[i][2] = three
    batchInput[i][3] = four
    batchTarget[i] = determinant([[batchInput[i][0], batchInput[i][1]], [batchInput[i][2], batchInput[i][3]]])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(batchInput, batchTarget,
    test_size=0.1, random_state=42)

bestHidden = (1, 1, 20)
for j in range(1, 20):
    for i in range(1, 20):
        regressor = skflow.TensorFlowDNNRegressor(hidden_units=[i, j], steps=10000, learning_rate=0.1)

        regressor.fit(X_train, y_train)

        score = metrics.mean_squared_error(regressor.predict(X_test), y_test)

        print('MSE of {0:d}, {1:d}: {2:f}'.format(i, j, score));
        if(score < bestHidden[2]):
            bestHidden = (i, j, score)

print('%d, %d with MSE of %f' % (bestHidden[0], bestHidden[1], bestHidden[2]))


# regressor = skflow.TensorFlowDNNRegressor(hidden_units=[11], steps=10000, learning_rate=0.1)

# regressor.fit(X_train, y_train)

# score = metrics.mean_squared_error(regressor.predict(X_test), y_test)

# print('MSE of {0:d}: {1:f}'.format(i, score));


while True:
    val1 = float(input("1: "))
    val2 = float(input("2: "))
    val3 = float(input("3: "))
    val4 = float(input("4: "))
    prediction = regressor.predict(np.array([[val1, val2, val3, val4]]))
    print("prediction: %f" % prediction[0][0])
    print("actual: %f" % determinant([[val1, val2], [val3, val4]]))