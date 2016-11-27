from six.moves import cPickle as pickle
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn, rnn_cell

NUM_STEPS = 2
INPUT_LEN = 2
NUM_HIDDEN = 2
BATCH_SIZE = 10


x = tf.placeholder("float", [BATCH_SIZE, NUM_STEPS, INPUT_LEN])
def RNN(x):
    x = tf.transpose(x,[1, 0, 2])
    x = tf.reshape(x, [-1, INPUT_LEN])
    x = tf.split(0, NUM_STEPS, x)
    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(NUM_HIDDEN)
    outputs, _ = rnn.rnn(rnn_cell, x, dtype=tf.float32)
    
    return outputs

outputs = RNN(x)

#########generate weights bias and inputs#########
weights =np.ones((INPUT_LEN + NUM_HIDDEN, NUM_HIDDEN))
bias = np.zeros((NUM_HIDDEN))
inputs = np.ones((BATCH_SIZE, NUM_STEPS, INPUT_LEN))


init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for var in tf.trainable_variables():
        if "Matrix" in var.name:
            sess.run(tf.assign(var, weights))
        if "bias" in var.name:
            sess.run(tf.assign(var, bias))
    output = sess.run(outputs, feed_dict = {x: inputs})    
    print type(output)
    print len(output)
    print output[0].shape
    print output[-1]



