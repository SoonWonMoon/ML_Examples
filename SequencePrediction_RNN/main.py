import numpy as np
import tensorflow as tf

# Hyper Constants
iterations = 1000
epoch_size = 100

class_num = 10
sequence_len = 50
batch_size = 200

# Generate Data
identity = np.eye(10)

def func_y(x):
    y0 = (x + 3) % 10

    y1 = np.roll(y0, 3)
    y1[:,0:3] = 0
    y2 = np.roll(y0, 5)
    y2[:,0:5] = 0
    y3 = np.roll(y0, 7)
    y3[:,0:7] = 0

    y = 0.3*identity[y1] - 0.5*identity[y2] + 0.7*identity[y3]

    return y

def generateBatch():
    x_data = np.array(np.random.choice(class_num,
        (batch_size, sequence_len)), dtype="int32")
    y_data = func_y(x_data)

    return (x_data, y_data)

# Define tf Variables
x = tf.placeholder(tf.int32, shape=[batch_size, sequence_len])
y = tf.placeholder(tf.float32, shape=[batch_size, sequence_len, class_num])
x_one_hot = tf.one_hot(x, class_num)

eps = tf.constant(0.01, shape=[batch_size, sequence_len])

#Define Network
gru = tf.contrib.cudnn_rnn.CudnnGRU(3, 50)
rnn_output, state = gru(tf.transpose(x_one_hot,[1,0,2]), initial_state=(tf.constant(0, shape=[3,batch_size,sequence_len], dtype=tf.float32),))

y_hat = tf.contrib.layers.fully_connected(tf.transpose(rnn_output, [1,0,2]), class_num, activation_fn = None)

square_error = tf.square(y - y_hat)
loss  = tf.reduce_mean(square_error)
train = tf.train.AdamOptimizer(learning_rate=0.002).minimize(loss)
acc   = tf.reduce_mean(tf.cast(tf.less(tf.sqrt(tf.reduce_mean(square_error, axis=-1)), eps), dtype=tf.float32))

# Run
if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(iterations):
            for j in range(epoch_size):
                x_data, y_data = generateBatch()

                sess.run(train, feed_dict={x:x_data, y:y_data})

            _loss, _acc = sess.run([loss, acc], feed_dict={x:x_data, y:y_data})
            print("Epoch : {:3d} | Loss : {:10.6f} | Acc : {:10.6f}".format(i, _loss, _acc))