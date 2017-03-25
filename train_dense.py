import numpy as np
import tensorflow as tf
import os

#function to create mini_batches for training
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]

    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)

    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        yield inputs[excerpt], targets[excerpt]

#function to implement a multilayer perceptron in Tensorflow
def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])     # Hidden layer with RELU activation
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])       # Hidden layer with RELU activation
    layer_2 = tf.nn.relu(layer_2)

    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']      # Output layer with linear activation
    return out_layer

#function to train the MLP
def train(trainX, trainY):
    # Parameters
    learning_rate = 0.001
    training_epochs = 40
    batch_size = 100
    display_step = 1

    # Network Parameters
    n_hidden_1 = 300 # 1st layer number of features
    n_hidden_2 = 300 # 2nd layer number of features
    n_input = 784
    n_classes = 10

    num_training_examples = trainX.shape[0]

    trainX = trainX.reshape(num_training_examples, n_input)
    trainY = trainY.reshape(num_training_examples, 1)

    trainY_actual = np.zeros([num_training_examples, n_classes])
    for pos, val in enumerate(trainY):
        trainY_actual[pos][val] = 1

    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])*np.sqrt(2./(n_input+n_hidden_1))),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])*np.sqrt(2./(n_hidden_1+n_hidden_2))),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes])*np.sqrt(2./(n_hidden_2+n_classes)))
    }
    biases = {
        'b1': tf.Variable(tf.zeros([n_hidden_1])),
        'b2': tf.Variable(tf.zeros([n_hidden_2])),
        'out': tf.Variable(tf.zeros([n_classes]))
    }

    pred = multilayer_perceptron(x, weights, biases)        # Construct the model

    cost = tf.reduce_mean(tf.losses.hinge_loss(logits=pred, labels=y))          # Define loss and optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()      # Initializing the variables

    total_number_of_batches = int(trainY_actual.shape[0] / batch_size)

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver({'weights[\'h1\']':weights['h1'], 'weights[\'h2\']':weights['h2'], 'weights[\'out\']':weights['out'], 'biases[\'b1\']':biases['b1'], 'biases[\'b2\']':biases['b2'], 'biases[\'out\']':biases['out']})

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = float(0.)

            for batch in iterate_minibatches (trainX, trainY_actual, batch_size, shuffle=True):
                batch_x, batch_y = batch
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})          # Run optimization op (backprop) and cost op (to get loss value)
                avg_cost = avg_cost + float(c) / float(total_number_of_batches)             # Compute average loss

            if epoch % display_step == 0:       # Display logs per epoch step
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

        save_path = saver.save(sess, "./weights/model1.ckpt")
        print("Optimization Finished.")

#function to test the trained MLP
def test(testX):
    '''
    This function reads the weight files and
    returns the predicted labels.
    The returned object is a 1-dimensional numpy array of
    length equal to the number of examples. The i-th element
    of the array contains the label of the i-th test
    example.
    '''
    # Network Parameters
    n_hidden_1 = 300 # 1st layer number of features
    n_hidden_2 = 300 # 2nd layer number of features
    n_input = 784
    n_classes = 10

    testX = testX.reshape(testX.shape[0], n_input)
    output_values = np.zeros(testX.shape[0])

    # tf Graph input
    x = tf.placeholder("float", [None, n_input])

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])*np.sqrt(2./(n_input+n_hidden_1))),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])*np.sqrt(2./(n_hidden_1+n_hidden_2))),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes])*np.sqrt(2./(n_hidden_2+n_classes)))
    }
    biases = {
        'b1': tf.Variable(tf.zeros([n_hidden_1])),
        'b2': tf.Variable(tf.zeros([n_hidden_2])),
        'out': tf.Variable(tf.zeros([n_classes]))
    }

    pred = multilayer_perceptron(x, weights, biases)        # Construct the model for MLP

    with tf.Session() as sess:
        saver = tf.train.Saver({'weights[\'h1\']':weights['h1'], 'weights[\'h2\']':weights['h2'], 'weights[\'out\']':weights['out'], 'biases[\'b1\']':biases['b1'], 'biases[\'b2\']':biases['b2'], 'biases[\'out\']':biases['out']})
        saver.restore(sess, "./weights/model1.ckpt")
        output_values = sess.run(pred, feed_dict={x:testX})

    print("Testing Finished.")

    predicted_labels = np.argmax(output_values, axis=1)

    return predicted_labels
