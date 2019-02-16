import numpy as np
import tensorflow as tf

# Hyper Constants
iterations = 10
batch_size = 500

# Load CIFAR-10
cifar10 = tf.keras.datasets.cifar10

(x_train, y_class_train), (x_test, y_class_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test  = x_test  / 255.0

y_class_train = y_class_train.reshape([-1])
y_class_test  = y_class_test.reshape([-1])

y_train = np.eye(10)[y_class_train]
y_test  = np.eye(10)[y_class_test]

n_train = 50000

# Define tf Variables
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, 10])

y_class = tf.placeholder(tf.int32, shape=[None])

# Define Sequence
t1 = tf.layers.conv2d(x, filters=40, kernel_size=(3,3), strides=(1,1), padding='same',  activation=tf.nn.relu)      # 32x32 * 40
t2 = tf.layers.max_pooling2d(t1, pool_size=(2,2), strides=(2,2), padding='valid')                                   # 16x16 * 40

t3 = tf.layers.conv2d(t2, filters=80, kernel_size=(3,3), strides=(1,1), padding='same', activation=tf.nn.relu)      # 16x16 * 80
t4 = tf.layers.max_pooling2d(t3, pool_size=(2,2), strides=(2,2), padding='valid')                                   # 8x8   * 80

t5 = tf.layers.conv2d(t4, filters=160, kernel_size=(3,3), strides=(1,1), padding='same', activation=tf.nn.relu)     # 8x8   * 160
t6 = tf.layers.max_pooling2d(t5, pool_size=(2,2), strides=(2,2), padding='valid')                                   # 4x4   * 160

t7 = tf.layers.conv2d(t6, filters=320, kernel_size=(3,3), strides=(1,1), padding='same', activation=tf.nn.relu)     # 4x4   * 320
t8 = tf.layers.max_pooling2d(t7, pool_size=(2,2), strides=(2,2), padding='valid')                                   # 2x2   * 320

t9 = tf.reshape(t8, [tf.shape(x)[0], 1280])

t10   = tf.contrib.layers.fully_connected(t9,  500, activation_fn=tf.nn.relu)
y_hat = tf.contrib.layers.fully_connected(t10, 10,  activation_fn=None)

loss  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_hat))
train = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss)
acc   = tf.reduce_mean(tf.cast(tf.equal(y_class, tf.argmax(y_hat, axis=1, output_type=tf.int32)), dtype=tf.float32))

# Run
if __name__ == "__main__":
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(iterations):
        masks = np.reshape(np.random.permutation(n_train), [n_train//batch_size, batch_size])
        for mask in masks:
            x_batch = x_train[mask]
            y_batch = y_train[mask]

            sess.run(train, feed_dict={x:x_batch, y:y_batch})
        train_loss, train_acc = sess.run([loss,acc], feed_dict={x:x_batch, y:y_batch, y_class:y_class_train[masks[-1]]})
        test_loss, test_acc = sess.run([loss,acc], feed_dict={x:x_test, y:y_test, y_class:y_class_test})
        print("Epoch : {:3d} | TrainLoss : {:10.6f} | TrainAcc : {:10.6f} | TestLoss : {:10.6f} | TestAcc : {:10.6f}".
            format(i+1, train_loss, train_acc, test_loss, test_acc))
