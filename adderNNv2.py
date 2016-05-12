import tensorflow as tf
import numpy as np
import random


def inference(x):
    W = tf.Variable(tf.truncated_normal([3, 41], stddev=0.1))
    x_augmented = tf.concat(0, [[1.0], x])
    x_packed = tf.pack([x_augmented])
    # tf.matmul requires equal rank tensors- multiplying a vector by a matrix is strictly forbidden
    logits = tf.matmul(x_packed, W)
    return logits


def loss(logits, labels):
    # The inputs logits and labels must have the same shape [batch_size, num_classes]
    # In this case the shape is [1, 11]
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss


def training(loss, learning_rate):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def number_to_onehot(number):
    onehot = [0] * 41
    onehot[number] = 1.0
    return [onehot]


def softmax_to_number(classifier):
    return tf.argmax(classifier, 1)


def demonstrate(sess, x, classifier):
    prediction = softmax_to_number(classifier)
    for i in range(4):
        for j in range(4):
            input_x = [random.choice(range(20)), random.choice(range(20))]
            output = prediction.eval(feed_dict={x: input_x})
            print("I think {} + {} = {}".format(input_x[0], input_x[1], output))
    print("")


x = tf.placeholder('float')
labels = tf.placeholder('float')
classifier = inference(x)
cost = loss(classifier, labels)
train_op = training(cost, learning_rate=0.01)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    while True:
        demonstrate(sess, x, classifier)
        for _ in range(10000):
            input_x = [random.choice(range(20)), random.choice(range(20))]
            expected_y = number_to_onehot(sum(input_x))
            logits = classifier.eval(feed_dict={x: input_x})
            cost.eval(feed_dict={x: input_x, labels: expected_y})
            sess.run([train_op, cost], feed_dict={x: input_x, labels: expected_y})