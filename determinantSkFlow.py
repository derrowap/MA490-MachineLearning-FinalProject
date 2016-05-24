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

def worker(units, out_q, unit_1, batchInput, batchTarget):
    """worker function"""
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(batchInput, batchTarget,
    test_size=0.1, random_state=42)

    regressor = skflow.TensorFlowDNNRegressor(hidden_units=units, steps=10000, learning_rate=0.1)

    regressor.fit(X_train, y_train)

    score = metrics.mean_squared_error(regressor.predict(X_test), y_test)

    unit_1.put(units[0])
    out_q.put(score)

    print('[%d] done' % units[0])
    return

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

# print(outs)
# print(outs_i)

# bestHidden = (1, 1, 20)
# for j in range(1, 20):
#     for i in range(1, 20):
#         regressor = skflow.TensorFlowDNNRegressor(hidden_units=[i, j], steps=10000, learning_rate=0.1)

#         regressor.fit(X_train, y_train)

#         score = metrics.mean_squared_error(regressor.predict(X_test), y_test)

#         print('MSE of {0:d}, {1:d}: {2:f}'.format(i, j, score));
#         if(score < bestHidden[2]):
#             bestHidden = (i, j, score)

# print('%d, %d with MSE of %f' % (bestHidden[0], bestHidden[1], bestHidden[2]))


# regressor = skflow.TensorFlowDNNRegressor(hidden_units=[2446], steps=100000, learning_rate=0.1)

# regressor.fit(batchInput, batchTarget)

# score = metrics.mean_squared_error(regressor.predict(batchInput), batchTarget)

# print('MSE: {0:f}'.format(score));


# while True:
#     val1 = float(input("1: "))
#     val2 = float(input("2: "))
#     val3 = float(input("3: "))
#     val4 = float(input("4: "))

#     start = time.clock()
#     prediction = regressor.predict(np.array([[val1, val2, val3, val4]]))
#     end = time.clock()
#     pred_time = end-start

#     start = time.clock()
#     determinant([[val1, val2], [val3, val4]])
#     end = time.clock()
#     det_time = end-start

#     print("prediction: %f" % prediction[0][0])
#     print("actual: %f" % determinant([[val1, val2], [val3, val4]]))



n_procs = 30
final_out = np.zeros(n_procs)
final_1 = np.zeros(n_procs)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(batchInput, batchTarget,
test_size=0.1, random_state=42)

procs = []
for j in range(n_procs):
    units = [(j+1) *100]

    regressor = skflow.TensorFlowDNNRegressor(hidden_units=units, steps=10000, learning_rate=0.1)

    regressor.fit(X_train, y_train)

    score = metrics.mean_squared_error(regressor.predict(X_test), y_test)

    final_out[j]=units[0]
    final_1[j]=score


np.savetxt('/home/sanderkd/Data/determinant_layer_by_100.csv', (final_1, final_out), delimiter=', ')


print('finished')