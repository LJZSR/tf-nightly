import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def generate(sample_size, num_classes, mean, cov, diff, regression):
    samples_per_class = int(sample_size/num_classes)
    
    X0 = np.random.multivariate_normal(mean, cov, samples_per_class)
    Y0 = np.zeros(samples_per_class)
    print(X0)
    for ci, d in enumerate(diff):
        X1 = np.random.multivariate_normal(mean+d, cov, samples_per_class)
        #print(ci,d)
        Y1 = (ci+1) * np.ones(samples_per_class)

        X0 = np.concatenate((X0,X1))
        Y0 = np.concatenate((Y0,Y1))

    #one-hot编码，将0转化为1 0
    if not regression:
        items = []
        for i in Y0:
            item = np.zeros(num_classes)
            item[np.int32(i)] = 1
            items.append(item)
        Y = np.vstack(items)
    #print(X0,Y0)
    #np.random.shuffle(X0)
    #np.random.shuffle(Y0)
    #print(Y)
    return X0, Y

np.random.seed(10)
num_classes = 3
input_dim = 2
lab_dim = num_classes
mean = np.random.randn(input_dim)
cov = np.eye(input_dim)
#print('mean=', mean, 'cov=', cov)
X, Y = generate(2000, num_classes, mean, cov, [[3.0,3.0],[3.0,0.0]], False)
aa = [np.argmax(l) for l in Y]
print(aa)
colors = ['r' if l == 0 else 'b' if l == 1 else 'y' for l in aa]
plt.scatter(X[:,0], X[:,1], c=colors)
plt.xlabel('Scaled age (in yrs)')
plt.ylabel('Tumber size (in cm)')
plt.show()

#定义占位符
input_features = tf.placeholder(tf.float32, [None, input_dim])
input_labels = tf.placeholder(tf.float32, [None, lab_dim])
#定义学习参数
W = tf.Variable(tf.random_normal([input_dim,lab_dim]), name='weight')
b = tf.Variable(tf.zeros([lab_dim]), name='bias')
output = tf.matmul(input_features, W) + b

z = tf.nn.softmax(output)

a1 = tf.argmax(z, axis=1)
b1 = tf.argmax(input_labels, axis=1)
err = tf.count_nonzero(a1-b1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=input_labels, logits=output)
loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(0.04)
train = optimizer.minimize(loss)
maxEpochs = 50
minibatchSize = 25

#启动Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(maxEpochs):
        sumerr = 0
        for i in range(np.int32(len(Y)/minibatchSize)):
            X1 = X[i*minibatchSize:(i+1)*minibatchSize,:]
            Y1 = Y[i*minibatchSize:(i+1)*minibatchSize,:]

            _, lossval, outputval, errval = sess.run([train,loss,output,err], feed_dict={input_features:X1,input_labels:Y1})
            sumerr += errval/minibatchSize
            
        print('Epoch:', '%04d' % (epoch+1), 'loss=', '{:.9f}'.format(lossval), 
            'err=', sumerr/np.int32(len(Y)/minibatchSize))

    train_X, train_Y = generate(200, num_classes, mean, cov, [[3.0,3.0],[3.0,0.0]], False)
    aa = [np.argmax(l) for l in train_Y]
    colors = ['r' if l==0 else 'b' if l==1 else 'y' for l in aa]
    plt.scatter(train_X[:,0], train_X[:,1], c=colors)
    plt.xlabel('Scaled age (in yrs)')
    plt.ylabel('Tumber size (in cm)')

    x = np.linspace(-1, 8, 200)
    for i in range(3):
        y = -(x*(sess.run(W)[0][i]/sess.run(W)[1][i])) - sess.run(b)[i]/sess.run(W)[1][i]
        plt.plot(x, y, label=str(i)+'th line', lw=3-i)
        
    plt.legend()
    plt.show()