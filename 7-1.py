import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def generate(sample_size, mean, cov, diff, regression):
    num_classes = 2
    samples_per_class = int(sample_size/2)
    
    X0 = np.random.multivariate_normal(mean, cov, samples_per_class)
    Y0 = np.zeros(samples_per_class)

    for ci, d in enumerate(diff):
        X1 = np.random.multivariate_normal(mean+d, cov, samples_per_class)
        Y1 = (ci+1) * np.ones(samples_per_class)

        X0 = np.concatenate((X0,X1))
        Y0 = np.concatenate((Y0,Y1))

    #one-hot编码，将0转化为1 0
    if not regression:
        class_ind = [Y==class_number for class_number in range(num_classes)]
        Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
    #print(X0,Y0)
    #np.random.shuffle(X0)
    #np.random.shuffle(Y0)
    #print(X0, Y0)
    return X0, Y0

np.random.seed(10)
num_classes = 2
mean = np.random.randn(num_classes)
cov = np.eye(num_classes)
X, Y = generate(1000, mean, cov, [3.0], True)
colors = ['r' if l == 0 else 'b' for l in Y[:]]
plt.scatter(X[:,0], X[:,1], c=colors)
plt.xlabel('Scaled age (in yrs)')
plt.ylabel('Tumber size (in cm)')
plt.show()
lab_dim = 1
input_dim = 2
#print(X, Y)
input_features = tf.placeholder(tf.float32, [None, input_dim])
input_labels = tf.placeholder(tf.float32, [None, lab_dim])
#定义学习参数
W = tf.Variable(tf.random_normal([input_dim, lab_dim]), name = 'weight')
b = tf.Variable(tf.zeros(lab_dim), name = 'bias')

output = tf.nn.sigmoid(tf.matmul(input_features, W ) + b)
cross_entropy = -(input_labels * tf.log(output) + (1 - input_labels) * tf.log(1 - output))
ser = tf.square(input_labels - output)
loss = tf.reduce_mean(cross_entropy)
err = tf.reduce_mean(ser)
optimizer = tf.train.AdamOptimizer(0.04)
train = optimizer.minimize(loss)
maxEpochs = 50
minibatchSize = 25

#启动Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sumerr = 0
    #向模型输入数据
    for epoch in range(maxEpochs):
        sumerr = 0
        for i in range(np.int32(len(Y)/minibatchSize)):
            X1 = X[i*minibatchSize:(i+1)*minibatchSize]
            Y1 = np.reshape(Y[i*minibatchSize:(i+1)*minibatchSize], [-1,1])
            #print(Y1)
            _, lossval, outputval, errval = sess.run([train, loss, output, err], 
            feed_dict={input_features: X1, input_labels: Y1})
            sumerr += errval
        
        print('Epoch:', '%04d' % (epoch+1), 'loss=', '{:.9f}'.format(lossval), 'err=', sumerr/np.int32(len(Y)/minibatchSize))
    
    train_X, train_Y = generate(100, mean, cov, [3.0], True)
    colors = ['r' if l==0 else 'b' for l in train_Y]
    plt.scatter(train_X[:,0], train_X[:,1], c=colors)
    x = np.linspace(-1, 8, 200)
    #print(sess.run(W)[0])
    #print(sess.run(W)[1])
    y = -(x*(sess.run(W)[0]/sess.run(W)[1]))-sess.run(b)/sess.run(W)[1]
    plt.plot(x, y, label = 'Fitted line')
    plt.legend()
    plt.show()
