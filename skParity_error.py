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

from trainingFunctions import evenParity

def worker(numSteps, q_score, q_steps, batchInput, batchTarget):
    """worker function"""
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(batchInput, batchTarget,
        test_size=0.1, random_state=42)

    NN = skflow.TensorFlowDNNClassifier(
        hidden_units=[16], steps=numSteps, n_classes=2)

    NN.fit(x_train, y_train)

    score = metrics.mean_squared_error(NN.predict(x_test), y_test)

    q_score.put(score)
    q_steps.put(numSteps / 10000)

    print('STEPS %d FINISHED.' % numSteps)
    return

x = np.zeros((65536, 16), dtype='int32')
y = np.zeros(65536, dtype='int32')
for i in range(65536):
    temp = bin(i)[2:].zfill(16)
    x[i] = [int(j) for j in temp]
    y[i] = evenParity(i)

if __name__ == '__main__':
    q_score = multiprocessing.Queue();
    q_steps = multiprocessing.Queue();
    n_procs = 50
    procs = []
    for i in range(n_procs):
        p = multiprocessing.Process(
            target=worker,
            args=((i+1)*10000, q_score, q_steps, x, y))
        procs.append(p)
        p.start()

    # Wait for all worker processes to finish
    for p in procs:
        p.join()

    # Collect all results into a single array
    final_scores = np.zeros(n_procs)
    for i in range(n_procs):
        index = int(q_steps.get()) - 1
        final_scores[index] = q_score.get()

    np.savetxt('/home/derrowap/Data/parity_error.csv', final_scores)
    print('finished')