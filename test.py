#coding:utf-8
# (tensorflow)$ python   用 Python API 写 TensorFlow 示例代码
import input_data

import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
w=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))

x=tf.placeholder(tf.float32,[None,784])
y=tf.nn.softmax(tf.matmul(x,w)+b)

y_=tf.placeholder(tf.float32,[None,10])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess.run(init)

for i in range(2):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  feed_dict=feed_dict={x: batch_xs, y_: batch_ys}
  sess.run(train_step, feed_dict)
  
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print "end"





















