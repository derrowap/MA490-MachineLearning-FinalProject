from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import time
import multiprocessing

import numpy as np
from sklearn import cross_validation, metrics
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

from tensorflow.contrib import skflow

from trainingFunctions import multiply

def worker(units, q_score, q_units, q_max, batchInput, batchTarget):
    """worker function"""
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(batchInput, batchTarget,
        test_size=0.1, random_state=42)

    NN = skflow.TensorFlowDNNRegressor(hidden_units=[units], steps=1000)

    NN.fit(x_train, y_train)

    score = metrics.mean_squared_error(NN.predict(x_test), y_test)
    correct = True
    count = 1
    while correct:
        pred = np.rint(NN.predict([[count, count]])[0])
        real = multiply(count, count)
        if pred == real:
            count += 1
        else:
            correct = False 
            print('Units %d failed at %d.' % (units, count))
    q_score.put(score)
    q_units.put(units)
    q_max.put(count)

    print('Units %d done.' % units)
    return

top = 10
x = np.zeros((top ** 2, 2))
y = np.zeros(top ** 2)
count = 0
for i in range(1, top+1):
    for j in range(1, top+1):
        x[count] = [i, j]
        y[count] = multiply(i, j)
        count += 1

if __name__ == '__main__':
    q_score = multiprocessing.Queue()
    q_units = multiprocessing.Queue()
    q_max = multiprocessing.Queue()
    n_procs = 5
    procs = []
    for i in range(n_procs):
        p = multiprocessing.Process(
            target=worker,
            args=(i+90, q_score, q_units, q_max, x, y))
        procs.append(p)
        p.start()

    # Wait for all worker processes to finish
    for p in procs:
        p.join()

    # Collect all results into a single array
    final_scores = np.zeros(n_procs)
    for i in range(n_procs):
        index = int(q_units.get()) - 90
        final_units[index] = index + 90
        final_scores[index] = q_score.get()
        final_max[index] = int(q_max.get())

    np.savetxt('/home/derrowap/Data/multiply_layer_scores.csv', (final_units, final_scores, final_max)))
    print('finished')