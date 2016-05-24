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


# def worker(units, out_q, unit_1, batchInput, batchTarget):
#     """worker function"""

#     X_train, X_test, y_train, y_test = cross_validation.train_test_split(batchInput, batchTarget,
#     test_size=0.5, random_state=42)

#     print(X_train[0])
#     print(y_train[0])

#     regressor = skflow.TensorFlowDNNRegressor(hidden_units=[100], steps=100, learning_rate=0.5)

#     regressor.fit(X_train, y_train)

#     print('fit')

#     score = metrics.mean_squared_error(regressor.predict(X_test), y_test)

#     unit_1.put(units[0])
#     out_q.put(score)

#     print('[%d] done' % (units[0]))
#     return


samp = 10000
matrix = 3

bIn = []
bOut = []

for i in range(samp):
    arr = []
    arr_1d = np.zeros(9)
    for j in range(matrix):
        inArr = []
        for k in range(matrix):
            num = np.random.randint(20)
            inArr.append(num)
            arr_1d[j*matrix+k] = num
        arr.append(inArr)
    bIn.append(arr_1d)
    bOut.append(determinant(arr))

bIn = np.array(bIn)
bOut = np.rint(np.array(bOut))


X_train, X_test, y_train, y_test = cross_validation.train_test_split(bIn, bOut,
    test_size=0.5, random_state=42)

regressor = skflow.TensorFlowDNNRegressor(hidden_units=[1000], steps=100000, learning_rate=0.5)

regressor.fit(X_train, y_train)

regressor.save('/home/sanderkd/Data/det3b3SkFlow')

pred = regressor.predict(X_test)
pred = np.reshape(pred, -1)
pred = np.rint(pred)
error = 1 - accuracy_score(y_test, pred)

print(error)

# if __name__ == '__main__':

#     n_procs = 30
#     unit_1 = multiprocessing.Queue();
#     out_q = multiprocessing.Queue();

#     procs = []
#     for j in range(n_procs):
#         p = multiprocessing.Process(
#             target=worker,
#             args=([((j+1)*100)], out_q, unit_1, bIn, bOut))
#         procs.append(p)
#         p.start()

#     # Wait for all worker processes to finish
#     for p in procs:
#         p.join()

#     print('shouldnt be here')
#     # Collect all results into a single array
#     final_out = np.zeros(n_procs)
#     final_1 = np.zeros(n_procs)
#     for j in range(n_procs):
#         final_out[j] = out_q.get()
#         final_1[j] = unit_1.get()
#     np.savetxt('/home/sanderkd/Data/determinant_3by3_by_100.csv', (final_1, final_out), delimiter=', ')

    
#     print('finished')