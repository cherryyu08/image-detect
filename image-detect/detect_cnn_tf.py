import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time


def load_dataset():

    descriptors = np.load("images/MICC-F2000_desc.npy")
    labels = np.load("images/MICC-F2000_label.npy")
    labels = labels.reshape(labels.shape[0], 1)

    return descriptors, labels


def load_dataset220():

    descriptors = np.load("images/MICC-F220_desc.npy")
    labels = np.load("images/MICC-F220_label.npy")
    labels = labels.reshape(labels.shape[0], 1)

    return descriptors, labels


X_train, Y_train = load_dataset()
print(X_train.shape, Y_train.shape)
X_test, Y_test = load_dataset220()
print(X_test.shape, Y_test.shape)


# create a list of random mini  batches from (X,Y)
def random_mini_batches(X, Y, mini_batch_size):
    # X---(m,Hi,Wi,Ci)
    # Y---(m,n_y) value 0 or 1
    m = X.shape[0]
    mini_batches = []

    # step 1: shuffle (X,Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # step 2: partition (shuffled_X,shuffled_Y).minus the end case.
    num_complete_minibatchs = int(m/mini_batch_size)

    for k in range(0, num_complete_minibatchs):
        mini_batch_X = shuffled_X[k*mini_batch_size:k*mini_batch_size+mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k*mini_batch_size:k*mini_batch_size+mini_batch_size, :]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # handling the end case (last mini_batch <mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatchs*mini_batch_size:m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatchs*mini_batch_size:m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


# 定义onehot数组
def convert_to_one_hot(Y, C):
    # 先将Y转换成一行数，再将数组中指定位置的数置为1
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    # Creates the placeholders for the tensorflow session
    X = tf.placeholder(tf.float32, shape=[None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, shape=[None, n_y])

    return X, Y


def initialize_parameters(lambd):
    W1 = tf.get_variable("W1", shape=[4, 4, 128, 64], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", shape=[2, 2, 64, 32], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W3 = tf.get_variable("W3", shape=[2, 2, 32, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W4 = tf.get_variable("W4", shape=[1, 1, 16, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    # add_to_collection函数将这个新生成变量的L2正则化损失加入集合
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambd)(W1))
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambd)(W2))
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambd)(W3))
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambd)(W4))

    parameters = {"W1": W1, "W2": W2, "W3": W3, "W4": W4}

    return parameters


# Implements the forward propagation for the model:
# CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
def forward_propagation(X, parameters):

    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']
    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    Z3 = tf.nn.conv2d(P2, W3, strides=[1, 1, 1, 1], padding='SAME')

    A3 = tf.nn.relu(Z3)

    P3 = tf.nn.max_pool(A3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    Z4 = tf.nn.conv2d(P3, W4, strides=[1, 1, 1, 1], padding='SAME')

    A4 = tf.nn.relu(Z4)

    P4 = tf.nn.max_pool(A4, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')

    # FLATTEN
    P4 = tf.contrib.layers.flatten(P4)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
    Z5 = tf.contrib.layers.fully_connected(P4, num_outputs=1, activation_fn=None)

    return Z5


def compute_cost(Z3, Y):

    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Z3, labels=Y))  # 计算均值

    # 将损失函数加入losses集合中
    tf.add_to_collection('losses', cost)
    # 将losses集合中的所有损失函数加起来得到最终的损失函数
    loss = tf.add_n(tf.get_collection('losses'))

    return loss


def predict(Z, X, Y, X_pre, Y_pre):

    predict_op = tf.nn.sigmoid(Z)
    predicts = predict_op.eval({X: X_pre, Y: Y_pre})
    m = predicts.shape[0]
    p = np.zeros((m, 1))
    y = Y.eval({Y: Y_pre})
    tp = 0.
    tn = 0.
    fp = 0.
    fn = 0.
    for i in range(0, m):
        if predicts[i, 0] > 0.5:
            p[i, 0] = 1
            if y[i, 0] == 1:
                tp += 1
            else:
                fp += 1
        else:
            p[i, 0] = 0
            if y[i, 0] == 1:
                fn += 1
            else:
                tn += 1
    prediction = np.sum((p == y) / m)

    return prediction, tp, tn, fp, fn


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.00061, num_epochs=100, minibatch_size=60, lambd=0.005, print_cost=True):

    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []
    train_acc = []
    test_acc = []

    # Create placeholders of the correct shape
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    # initialize parameters
    parameters = initialize_parameters(lambd)

    # forward propagation : build the forward propagation in the tensorflow graph
    Z = forward_propagation(X, parameters)

    # cost function
    cost = compute_cost(Z, Y)

    # backpropagation
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # initial all the variables globally
    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch

                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                minibatch_cost += temp_cost / num_minibatches

            train_accuracy, _, _, _, _ = predict(Z, X, Y, X_train, Y_train)
            test_accuracy, train_tp, train_tn, train_fp, train_fn = predict(Z, X, Y, X_test, Y_test)
            train_tpr = train_tp / (train_tp + train_fn)
            train_fpr = train_fp / (train_fp + train_tn)

            # print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
                print("train_accuracy after epoch %i: %f" % (epoch, train_accuracy))
                print("test_accuracy after epoch %i: %f" % (epoch, test_accuracy))
                print("TP,TN,FP,FN,TPR,FPR : "
                      "%f %f %f %f %f %f" % (train_tp, train_tn, train_fp, train_fn, train_tpr, train_fpr))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
                train_acc.append(train_accuracy)
                test_acc.append(test_accuracy)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()

        plt.plot(np.squeeze(train_acc))
        plt.ylabel('train_acc')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()

        plt.plot(np.squeeze(test_acc))
        plt.ylabel('test_acc')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate = " + str(learning_rate))

        plt.show()

        train_accuracy, _, _, _, _ = predict(Z, X, Y, X_train, Y_train)
        test_accuracy, _, _, _, _ = predict(Z, X, Y, X_test, Y_test)

        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        return train_accuracy, test_accuracy, parameters


t = time.time()
# test
_, _, parameters = model(X_train, Y_train, X_test, Y_test)

print(str(time.time()-t)+"s")
