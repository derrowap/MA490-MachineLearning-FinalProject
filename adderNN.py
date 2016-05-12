# start a session which transfers data to a C++ envirnment where it is optimized for perfomance
import tensorflow as tf
import random as rn
import trainingFunctions as funcs
sess = tf.InteractiveSession()

# two functions to initialize weight and bias
# weight is initiated with slight noise for symmetry breaking
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# bias is initialized with slight positive to avoid dead neurons
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# placeholders that we'll ask tensorflow to fill later
# shape lets us know that it is a 2d tensor that has a first dimension of any size, and a second dimension of size 784 (for x)
x = tf.placeholder(tf.float32, shape=[None, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

# Variable is a value that lives in TensorFlow's computation graph
W = tf.Variable(tf.zeros([1,1])) # W is 1x1 matrix because 1 input and 1 output
b = tf.Variable(tf.zeros([1])) # b is 1-dimensional vector (we have 10 classes)

# initialize variables in session
sess.run(tf.initialize_all_variables())

# multiply input image by weight matrix and add the bias
y = tf.nn.sigmoid(tf.matmul(x,W) + b)

# cost function (we try to minimize) is cross-entropy between the target and model's prediction
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))

# add new operation to computation gaph
# train_step will apply gradient descent updates to parameters 
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# repeatedly calling train_step will train the model
for i in range(100):
    init = 100
    valArr = [0] * init
    ansArr = [0] * init
    for k in range(init):
        ranVal = rn.randint(1, 100)
        ans = funcs.adder(ranVal)
        valArr[k] = [ranVal]
        ansArr[k] = [ans]
    train_step.run(feed_dict={x: valArr, y_: ansArr}) #feed_dict will fill our placeholders

# checks if the predicted label and actualy label are equal
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# changed [True, False, True, True] to [1, 0, 1, 1] and takes mean (probability)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# init = 100
# valArr = [0] * init
# ansArr = [0] * init
# for k in range(init):
#         ranVal = rn.randint(1, 10000)
#         ans = funcs.adder(ranVal)
#         valArr[k] = [ranVal]
#         ansArr[k] = [ans]
# print(accuracy.eval(feed_dict={x: valArr, y_: ansArr}))

while True:
    val = int(input("Enter val to add: "))
    # prediction = tf.argmax(y, 1)
    classification = y.eval(feed_dict = {x: [[val]]})
    print(classification)