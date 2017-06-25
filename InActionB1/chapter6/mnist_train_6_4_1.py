# -*- coding:utf-8 -*-
import os
import tensorflow as tf
import mnist_inference_6_4_1
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from tensorflow.contrib import layers

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARZTION_RATE = 0.0001
TRAINING_STEPS = 60000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "/home/soso/PycharmProjects/MODEL_SAVE/"
MODEL_NAME = "model_6_4_1.ckpt"

def train(mnist):
    x = tf.placeholder(tf.float32,[BATCH_SIZE,mnist_inference_6_4_1.IMAGE_SIZE,mnist_inference_6_4_1.IMAGE_SIZE,mnist_inference_6_4_1.NUM_CHANNELS],name='x-input')
    y_ = tf.placeholder(tf.float32,[None,mnist_inference_6_4_1.OUTPUT_NODE],name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARZTION_RATE)
    y = mnist_inference_6_4_1.inference(x,True,regularizer)

    global_step = tf.Variable(0,trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples / BATCH_SIZE,LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    train_op = tf.group(train_step, variable_averages_op)

    saver = tf.train.Saver()

    net = layers.conv2d(input_data,32,[3,3])


    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs,(BATCH_SIZE,mnist_inference_6_4_1.IMAGE_SIZE,mnist_inference_6_4_1.IMAGE_SIZE,mnist_inference_6_4_1.NUM_CHANNELS))
            _,loss_value,step = sess.run([train_op,loss,global_step],feed_dict={x:reshaped_xs,y_:ys})

            if i % 100 == 0:
                print("After %d training step(s), loss is %g" %(step,loss_value))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

def main(argv = None):
    mnist = input_data.read_data_sets("/home/soso/MNIST_data",one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
