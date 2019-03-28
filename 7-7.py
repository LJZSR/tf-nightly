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
            item[int(i)] = 1
            items.append(item)
        Y0 = np.vstack(items)
    #print(X0,Y0)
    #np.random.shuffle(X0)
    #np.random.shuffle(Y0)
    #print(Y0)
    return X0, Y0

np.random.seed(10)
input_dim = 2
num_classes = 4
mean = np.random.randn(input_dim)
cov = np.eye(input_dim)
X, Y = generate(320, num_classes, mean, cov, [[3.0,0],[3.0,3.0],[0,3.0]], True)
Y = Y % 2
print(Y)

xr = []
xb = []
for (l,k) in zip(Y[:], X[:]):
    if l == 0.0:
        xr.append([k[0], k[1]])
    else:
        xb.append([k[0], k[1]])
#print(xr)
xr = np.array(xr)
xb = np.array(xb)
#print(xr)
plt.scatter(xr[:,0], xr[:,1], c='r', marker='+')
plt.scatter(xb[:,0], xr[:,1], c='b', marker='o')
plt.show()
X = np.array(X)
Y = np.array(Y)
Y = np.reshape(Y, [-1,1])
print(Y)

learning_rate = 1e-4
input_dim  = 2 #输入层节点个数
n_label = 1 
n_hidden = 200 #隐藏层节点个数

x = tf.placeholder(tf.float32, [None, input_dim])
y = tf.placeholder(tf.float32, [None, n_label])
weights = {
    'h1': tf.Variable(tf.random_normal([input_dim,n_hidden], stddev=0.1)),
    'h2': tf.Variable(tf.random_normal([n_hidden,n_label], stddev=0.1)),
}
biases = {
    'b1': tf.Variable(tf.zeros([n_hidden])),
    'b2': tf.Variable(tf.zeros([n_label])),
}
layer_1 = tf.nn.relu(tf.add(tf.matmul(x,weights['h1']),biases['b1']))
y_pred = tf.nn.tanh(tf.add(tf.matmul(layer_1,weights['h2']),biases['b2']))

#误差正则化
reg = 0.01
loss = tf.reduce_mean((y_pred-y)**2) + tf.nn.l2_loss(weights['h1'])*reg + tf.nn.l2_loss(weights['h2'])*reg

train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

#加载Session
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

#训练
for i in range(20000):
    lossval, _ = sess.run([loss, train_step], feed_dict={x:X,y:Y})
    print('Step: ', i, 'Current loss:', lossval)



#模型预测结果
xTrain, yTrain = generate(12, num_classes, mean, cov, [[3.0,0],[3.0,3.0],[0,3.0]],True)
yTrain = yTrain % 2
xr = []
xb = []
for (l,k) in zip(yTrain[:], xTrain[:]):
    if l == 0.0:
        xr.append([k[0], k[1]])
    else:
        xb.append([k[0], k[1]])

xr = np.array(xr)
xb = np.array(xb)
plt.scatter(xr[:,0], xr[:,1], c='r', marker='+')
plt.scatter(xb[:,0], xb[:,1], c='b', marker='o')
yTrain = np.reshape(yTrain, [-1,1])
print('loss:\n', sess.run(loss, feed_dict={x:xTrain, y:yTrain}))

nb_of_xs = 200
xs1 = np.linspace(-1, 8, nb_of_xs)
xs2 = np.linspace(-1, 8, nb_of_xs)
xx, yy = np.meshgrid(xs1, xs2)
#初始和填充classification plane
classification_plane = np.zeros((nb_of_xs, nb_of_xs))
for i in range(nb_of_xs):
    for j in range(nb_of_xs):
        classification_plane[i,j] = sess.run(y_pred, feed_dict={x:[[ xx[i,j],yy[i,j] ]]})
        classification_plane[i,j] = int(classification_plane[i,j]+0.5)
    
#创建一个color map用来显示每一个格子的分类颜色
cmap = ListedColormap([
    ColorConverter.to_rgba('r', alpha=0.3),
    ColorConverter.to_rgba('b', alpha=0.3)
])
plt.contourf(xx, yy, classification_plane, cmap=cmap)
plt.show()

