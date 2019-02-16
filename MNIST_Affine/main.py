import numpy as np
import tensorflow as tf

# Hyper Constants
iterations = 20
epoch_size = 500
batch_size = 500

# Load MNIST
mnist = tf.keras.datasets.mnist
one_hot_vectors = np.eye(10)

(x_train, y_train_class), (x_test, y_test_class) = mnist.load_data()
x_train = x_train.reshape((60000,784))
x_test  = x_test.reshape((10000,784))
x_train = x_train / 255.0
x_test  = x_test  / 255.0

y_train = one_hot_vectors[y_train_class]
y_test  = one_hot_vectors[y_test_class]

n_train = 60000

# Define tf Variables
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10 ])

W1 = tf.Variable(tf.random_normal([784,100], stddev=1/np.sqrt(784)), name="W1")
W2 = tf.Variable(tf.random_normal([100,100], stddev=1/np.sqrt(100)), name="W2")
W3 = tf.Variable(tf.random_normal([100, 10], stddev=1/np.sqrt(100)), name="W3")

B1 = tf.Variable(tf.zeros([100]), name="B1")
B2 = tf.Variable(tf.zeros([100]), name="B2")
B3 = tf.Variable(tf.zeros([10]),  name="B3")

y_class = tf.placeholder(tf.int32, shape=[None])

# Define Sequence
y1 = tf.sigmoid(tf.matmul(x,  W1) + B1)
y2 = tf.sigmoid(tf.matmul(y1, W2) + B2)
y_hat = tf.nn.softmax(tf.matmul(y2, W3) + B3)

loss  = tf.reduce_mean(tf.square(y - y_hat))
train = tf.train.RMSPropOptimizer(learning_rate=0.0005).minimize(loss)
acc   = tf.reduce_mean(tf.cast(tf.equal(y_class, tf.argmax(y_hat, axis=1, output_type=tf.int32)), dtype=tf.float32))

# Run
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(iterations):
        for j in range(epoch_size):
            batch_mask = np.random.choice(n_train, batch_size)
            x_batch = x_train[batch_mask]
            y_batch = y_train[batch_mask]

            sess.run(train, feed_dict={x:x_batch, y:y_batch})

        y_class_batch = y_train_class[batch_mask]

        train_loss, train_acc = sess.run([loss, acc],
            feed_dict={x:x_batch, y:y_batch, y_class:y_class_batch})
        test_loss,  test_acc  = sess.run([loss, acc],
            feed_dict={x:x_test,  y:y_test,  y_class:y_test_class})
        print(train_loss, train_acc, test_loss, test_acc)