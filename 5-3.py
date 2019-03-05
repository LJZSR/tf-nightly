import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
import pylab

tf.reset_default_graph()
#定义占位符
x = tf.placeholder(tf.float32, [None, 784]) #mnist数据集的维度是28*28=784
y = tf.placeholder(tf.float32, [None, 10]) #数字0-9，共10个类别
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([10]))
pred = tf.nn.softmax(tf.matmul(x, W) + b) #softmax分类

#损失函数
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

#定义参数
learning_rate = 0.01

#使用梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

traing_epochs = 25
batch_size = 100
display_step = 1

saver = tf.train.Saver()
model_path = 'log/521model.ckpt'
#启动Session
print('Staring 2nd session...')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #恢复模型变量
    saver.restore(sess, model_path)

    #测试model
    correction_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    #计算准确率
    accuracy = tf.reduce_mean(tf.cast(correction_prediction, tf.float32))
    print('Accuracy:', accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))

    output = tf.argmax(pred, 1)
    batch_xs, batch_ys = mnist.train.next_batch(2)
    outputval, predv = sess.run([output, pred], feed_dict={x: batch_xs})
    print(outputval, predv, batch_ys)

    im = batch_xs[0]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()

    im = batch_xs[1]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()
    