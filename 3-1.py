# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3

plt.plot(train_X, train_Y, 'ro', label='Origin data')
plt.legend()
plt.show()

X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')

z = tf.multiply(X, W) + b

cost = tf.reduce_mean(tf.square(Y - z))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

trainging_epochs = 20
display_step = 2

with tf.Session() as sess:
    sess.run(init)
    plotdata = {'batchsize':[], 'loss':[]}
    def moving_average(a, w=10):
        if len(a) < w:
            return a[:]
        return [val if idx < w else sum(a[idx-w:idx])/w for idx, val in enumerate(a)]

    for epoch in range(trainging_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        if epoch % display_step == 0:
            loss = sess.run(cost,feed_dict={X:train_X,Y:train_Y})
            print('Epoch:', epoch+1, 'cost=', loss, 'W=', sess.run(W), 'b=', sess.run(b))
            if not (loss=='NA'):
                plotdata['batchsize'].append(epoch)
                plotdata['loss'].append(loss)

    print('Finished!')
    print('cost=', sess.run(cost, feed_dict={X:train_X,Y:train_Y}), 'W=', sess.run(W), 'b=', sess.run(b))

    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W)*train_X + sess.run(b), label='Fittedline')
    plt.legend()
    plt.show()

    plotdata['avgloss'] = moving_average(plotdata['loss'])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata['batchsize'], plotdata['avgloss'], 'b--' )
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss')
    plt.show()