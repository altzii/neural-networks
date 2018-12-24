import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Read data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

image_size = 28
labels_size = 10
learning_rate = 0.5
steps_number = 2000
batch_size = 100
hidden_size = 1024
dropout_rate = 0.5

# Define placeholders
x = tf.placeholder(tf.float32, [None, image_size * image_size])
y_ = tf.placeholder(tf.float32, [None, labels_size])

# Variables for the hidden layer
W_h = tf.Variable(tf.truncated_normal(shape=[image_size * image_size, hidden_size], stddev=0.1))
b_h = tf.Variable(tf.truncated_normal(shape=[hidden_size], stddev=0.1))

# Hidden layer with reLU activation function
hidden = tf.layers.dropout(tf.nn.relu(tf.matmul(x, W_h) + b_h))

# Variables for the output layer
W = tf.Variable(tf.zeros([hidden_size, labels_size]))
b = tf.Variable(tf.zeros([labels_size]))

# Connect hidden to the output layer
y = tf.nn.softmax(tf.matmul(hidden, W) + b)

keep_probability = tf.placeholder(tf.float32)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

init = tf.global_variables_initializer()

# Run the training
sess = tf.Session()
sess.run(init)

for i in range(steps_number):
    # Get the next batch
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    # Run the training step
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_probability: dropout_rate})

# Accuracy calculation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Evaluate on the test set and print
print("Accuracy: %s" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
