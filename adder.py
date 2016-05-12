from tensorflow.contrib import skflow
import numpy as np

regressor = skflow.TensorFlowEstimator.restore('/home/sanderkd/Data/adderSkFlow')

while True:
    val = int(input("Enter val to add: "))
    prediction = regressor.predict(np.array([[val]]))
    print("prediction: %d" % int(prediction[0][0]))