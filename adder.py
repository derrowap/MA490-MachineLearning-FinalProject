from tensorflow.contrib import skflow
import numpy as np
import time

regressor = skflow.TensorFlowEstimator.restore('/home/sanderkd/Data/adderSkFlow')

while True:
    val = int(input("Enter val to add: "))
    start = time.clock()
    prediction = regressor.predict(np.array([[val]]))
    end = time.clock()
    time1 = end-start

    start = time.clock()
    val = val + 42
    end = time.clock()
    time2 = end-start
    print("%f" % time2)
    print("prediction: %d took %f seconds" % ((int(prediction[0][0])), time1))
    print("took %f times longer" % (time1/time2))