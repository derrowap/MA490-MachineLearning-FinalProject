from tensorflow.contrib import skflow
import numpy as np
import time
from trainingFunctions import determinant

regressor = skflow.TensorFlowEstimator.restore('/home/sanderkd/Data/det3b3SkFlow')

while True:
    val1 = float(input("1: "))
    val2 = float(input("2: "))
    val3 = float(input("3: "))
    val4 = float(input("4: "))
    val5 = float(input("4: "))
    val6 = float(input("4: "))
    val7 = float(input("4: "))
    val8 = float(input("4: "))
    val9 = float(input("4: "))
    start = time.clock()
    prediction = regressor.predict(np.array([[val1, val2, val3, val4, val5, val6, val7, val8, val9]]))
    end = time.clock()
    time1 = end-start

    start = time.clock()
    val = determinant([[val1, val2, val3], [val4, val5, val6], [val7, val8, val9]])
    end = time.clock()
    time2 = end-start
    print("%f" % time2)
    print("prediction: %d took %f seconds" % (((prediction[0][0])), time1))
    print("took %f times longer" % (time1/time2))