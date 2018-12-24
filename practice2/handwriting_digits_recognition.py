import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Read data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

image_size = 28
labels_size = 10
learning_rate = 0.5
steps_number = 1000
batch_size = 100

# Define placeholders
x = tf.placeholder(tf.float32, [None, image_size * image_size])
y_ = tf.placeholder(tf.float32, [None, labels_size])

W = tf.Variable(tf.zeros([image_size * image_size, labels_size]))
b = tf.Variable(tf.zeros([labels_size]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

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
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Accuracy calculation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Evaluate on the test set and print
print("Accuracy: %s" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
