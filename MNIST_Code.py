import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

learning_rate = 0.0001
batch_size = 100
update_step = 10

layer_1_nodes = 500
layer_2_nodes = 500
layer_3_nodes = 500
output_nodes = 10

network_input = tf.placeholder(tf.float32, [None, 784])
target_output = tf.placeholder(tf.float32, [None, output_nodes])

layer_1 = tf.Variable(tf.random_normal([784, layer_1_nodes]))
layer_1_bias = tf.Variable(tf.random_normal([layer_1_nodes]))
layer_2 = tf.Variable(tf.random_normal([layer_1_nodes, layer_2_nodes]))
layer_2_bias = tf.Variable(tf.random_normal([layer_2_nodes]))
layer_3 = tf.Variable(tf.random_normal([layer_2_nodes, layer_3_nodes]))
layer_3_bias = tf.Variable(tf.random_normal([layer_3_nodes]))
out_layer = tf.Variable(tf.random_normal([layer_3_nodes, output_nodes]))
out_layer_bias = tf.Variable(tf.random_normal([output_nodes]))

l1_output = tf.nn.relu(tf.matmul(network_input, layer_1) + layer_1_bias)
l2_output = tf.nn.relu(tf.matmul(l1_output, layer_2) + layer_2_bias)
l3_output = tf.nn.relu(tf.matmul(l2_output, layer_3) + layer_3_bias)
ntwk_output_1 = tf.matmul(l3_output, out_layer) + out_layer_bias
ntwk_output_2 = tf.nn.softmax(ntwk_output_1)
cf = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ntwk_output_1, labels=target_output))
ts = tf.train.GradientDescentOptimizer(learning_rate).minimize(cf)
cp = tf.equal(tf.argmax(ntwk_output_2, 1), tf.argmax(target_output, 1))
acc = tf.reduce_mean(tf.cast(cp, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    num_epochs = 10
    for epoch in range(hm_epochs):
        total_cost = 0
        for _ in range(int(mnist.train.num_examples / batch_size)):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            t, c = sess.run([ts, cf], feed_dict={network_input: batch_x, target_output: batch_y})
            # t, c = sess.run([optimizer, cross_entropy], feed_dict={network_input: batch_x, target_output: batch_y})
            total_cost += c
        print('Epoch', epoch, 'completed out of', num_epochs, 'loss:', total_cost)
        #print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
    print('Accuracy:', acc.eval({network_input: mnist.test.images,target_output: mnist.test.labels}))

