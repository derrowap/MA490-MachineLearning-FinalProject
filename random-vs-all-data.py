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

from trainingFunctions import adder

def worker(steps, out_q, unit_1, batchInput, batchTarget):
    """worker function"""

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(batchInput, batchTarget,
    test_size=0.1, random_state=42)

    regressor = skflow.TensorFlowDNNRegressor(hidden_units=[2], steps=steps, learning_rate=0.1)

    regressor.fit(X_train, y_train)

    score = metrics.mean_squared_error(regressor.predict(X_test), y_test)

    unit_1.put(steps)
    out_q.put(score)

    print('%d done' % steps)
    return

if __name__ == '__main__':

    def runAll():
        procs = []
        for i in range(n_procs):
            p = multiprocessing.Process(
                target=worker,
                args=((i+1)*100, out_q, unit_1, batchInput, batchTarget))
            procs.append(p)
            p.start()

        # Wait for all worker processes to finish
        for p in procs:
            p.join()

        return

    for j in range(5):
        ex = 100
        batchInput = np.zeros(ex)
        batchTarget = np.zeros(ex)
        n_procs = 100
        unit_1 = multiprocessing.Queue();
        out_q = multiprocessing.Queue();

        for i in range(ex):
            batchInput[i] = np.random.randint(100)
            batchTarget[i] = adder(batchInput[i])
            
        procs = []
        for i in range(n_procs):
            p = multiprocessing.Process(
                target=worker,
                args=((i+1)*100, out_q, unit_1, batchInput, batchTarget))
            procs.append(p)
            p.start()

        # Wait for all worker processes to finish
        for p in procs:
            p.join()

        # Collect all results into a single array
        final_out = np.zeros(n_procs)
        final_1 = np.zeros(n_procs)

        for i in range(n_procs):
            final_out[i] = out_q.get()
            final_1[i] = unit_1.get()

        file_path = '/home/sanderkd/Data/adder_random_'+str(j)+'_data.csv'
        np.savetxt(file_path, (final_1, final_out), delimiter=', ')
    
    print('finished')