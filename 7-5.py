import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
import pylab

#定义参数
learning_rate = 0.001
traing_epochs = 25
batch_size = 100
display_step = 1

#设置网络模型参数
n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes = 10

#定义占位符
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

#创建model
def multilayer_perceptron(x, weights, biases):
    #第一层隐含层
    layer_1 = tf.add(tf.matmul(x,weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    #第二隐含层
    layer_2 = tf.add(tf.matmul(layer_1,weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    #输出层
    out_layer = tf.add(tf.matmul(layer_2,weights['out']), biases['out'])
    return out_layer

#学习参数
weights = {
    'h1': tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2,n_classes])),
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes])),
}

#输出值
pred = multilayer_perceptron(x, weights, biases)

#定义loss和优化器
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#启动Session
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    #启动循环，开始训练
    for epoch in range(traing_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        #循环所有数据集
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            #运行优化器
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
            #计算loss平均值
            avg_cost += c / batch_size
        
        #显示训练中详细信息
        if (epoch+1) % display_step == 0:
            print('Epoch:', '%04d' % (epoch+1), 'cost=', '{:.9f}'.format(avg_cost))
    
    print('Finished!')

    #测试model
    correction_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    #计算准确率
    accuracy = tf.reduce_mean(tf.cast(correction_prediction, tf.float32))
    print('Accuracy:', accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))


