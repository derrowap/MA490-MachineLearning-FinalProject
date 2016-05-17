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

# hiddenSizeFirst = 10
# hiddenSizeSecond = 10
# hiddenArr = np.zeros((hiddenSizeFirst*hiddenSizeSecond, 4))
# loop = 0

# for firstLayer in range(1, hiddenSizeFirst+1):
#     for secondLayer in range(1, hiddenSizeSecond+1):
#         #generate data
batchSize = 10000
batchInput = np.zeros(batchSize)
batchTarget = np.zeros(batchSize)
i = 0
for j in range(batchSize):
    i+=1
    # batchInput[j] = np.random.randint(100)
    # batchTarget[j] = adder(batchInput[j])
    batchInput[j] = i
    batchTarget[j] = adder(i)
    if(i == 1000):
        i = 0


# Split dataset into train / test
X_train, X_test, y_train, y_test = cross_validation.train_test_split(batchInput, batchTarget,
    test_size=0.1, random_state=42)


# Scale data (training set) to 0 mean and unit standard deviation.
# scaler = preprocessing.StandardScaler()
# X_train = scaler.fit_transform(X_train)


# Build 2 layer fully connected DNN with 10, 10 units respectively.
# regressor = skflow.TensorFlowEstimator.restore('/home/sanderkd/Data/adderSkFlow')
# regressor = skflow.TensorFlowDNNRegressor(hidden_units=[4, 9], steps=1000000, learning_rate=0.1)
regressor = skflow.TensorFlowLinearRegression()

# Fit
start = time.clock()
regressor.fit(X_train, y_train)
end = time.clock()
time = end-start
print('Took %0.5f seconds to fit' % time)

#save
regressor.save('/home/sanderkd/Data/adderSkFlow')


# Predict and score
score = metrics.mean_squared_error(regressor.predict(X_test), y_test)

# print('MSE: {0:f}'.format(score))

# pred = regressor.predict(X_train)
# pred = np.reshape(pred, -1)
# pred = np.rint(pred)
# error_train = 1 - accuracy_score(y_train, pred)

# pred = regressor.predict(X_test)
# pred = np.reshape(pred, -1)
# pred = np.rint(pred)
# error_test = 1 - accuracy_score(y_test, pred)

# print('\nTraining error = %0.3f    testing error = %0.3f'
#     % (error_train, error_test))

i = 1
while(adder(i) == round(regressor.predict(np.array([[i]]))[0][0])):
    i += 1

print('Adder first failed at %d' % i)
# hiddenArr[loop][0] = firstLayer
# hiddenArr[loop][1] = secondLayer
# hiddenArr[loop][2] = score
# hiddenArr[loop][3] = i

# loop += 1

# for i in range(loop-1):
#     print('layer [%d, %d], MSE %f, failed at %d' % (hiddenArr[i][0], hiddenArr[i][1], hiddenArr[i][2], hiddenArr[i][3]))

# while True:
#     val = int(input("Enter val to add: "))
#     prediction = regressor.predict(np.array([[val]]))
#     print("prediction: %d" % round(prediction[0][0]))