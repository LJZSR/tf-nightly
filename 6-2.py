import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/')
import pylab

tf.reset_default_graph()
#定义占位符
x = tf.placeholder(tf.float32, [None, 784]) #mnist数据集的维度是28*28=784
y = tf.placeholder(tf.int32) #数字0-9，共10个类别
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([10]))
pred = tf.matmul(x, W) + b 
#损失函数
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred))

#定义参数
learning_rate = 0.01

#使用梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

traing_epochs = 25
batch_size = 100
display_step = 1

#saver = tf.train.Saver()
#model_path = 'log/521model.ckpt'
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
    correction_prediction = tf.equal(tf.argmax(pred,1), tf.cast(y,dtype=tf.int64))
    #计算准确率
    accuracy = tf.reduce_mean(tf.cast(correction_prediction, tf.float32))
    print('Accuracy:', accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))

    #保存模型
    #save_path = saver.save(sess, model_path)
    #print('Model saved in files: %s' %save_path)