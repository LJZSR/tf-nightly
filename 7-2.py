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
        print(ci,d)
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
        Y = np.vstack(items)
    #print(X0,Y0)
    #np.random.shuffle(X0)
    #np.random.shuffle(Y0)
    print(Y)
    return X0, Y

np.random.seed(10)
input_dim = 2
num_classes = 3
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