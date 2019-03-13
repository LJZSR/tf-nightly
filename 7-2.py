import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, ColorConverter
import tensorflow as tf

def generate(sample_size, num_classes, mean, cov, diff, regression):
    samples_per_class = int(sample_size/num_classes)
    
    X0 = np.random.multivariate_normal(mean, cov, samples_per_class)
    Y0 = np.zeros(samples_per_class)
    #print(X0)
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

    nb_of_xs = 200
    xs1 = np.linspace(-1, 8, num=nb_of_xs)
    xs2 = np.linspace(-1, 8, num=nb_of_xs)
    xx, yy = np.meshgrid(xs1, xs2) #创建网络
    #初始化classification_plane
    classisfication_plane = np.zeros((nb_of_xs, nb_of_xs))
    for i in range(nb_of_xs):
        for j in range(nb_of_xs):

            classisfication_plane[i,j] = sess.run(a1, feed_dict={input_features:[[xx[i,j],yy[i,j]]]})

    #创建color map 用于显示
    cmap = ListedColormap([
        ColorConverter.to_rgba('r', alpha=0.3),
        ColorConverter.to_rgba('b', alpha=0.3),
        ColorConverter.to_rgba('y', alpha=0.3),
    ])
    #图示各个样品边界
    plt.contourf(xx, yy, classisfication_plane, cmap=cmap)
    plt.show()