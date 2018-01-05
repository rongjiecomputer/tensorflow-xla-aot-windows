import argparse
import sys

import tensorflow as tf

from tensorflow.core.protobuf import saver_pb2
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import ops
from tensorflow.python.training import saver as saver_lib

GRAPH_FILE = "mnist.pb"
CKPT_FILE = "mnist.ckpt"

def main(_):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  g = ops.Graph()
  with g.as_default():
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y_')

    w = tf.Variable(tf.zeros([784, 10]), name='w')
    b = tf.Variable(tf.zeros([10]), name='b')
    y = tf.add(tf.matmul(x, w), b, name='y')

    with open(GRAPH_FILE, "wb") as f:
      f.write(g.as_graph_def().SerializeToString())

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V1)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      train_loops = 1000
      for i in range(train_loops):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

      saver.save(sess, CKPT_FILE)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/mnist')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)