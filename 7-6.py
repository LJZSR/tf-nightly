import numpy as np
import matplotlib.pyplot as plt

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
print(xr)
xr = np.array(xr)
xb = np.array(xb)
print(xr)
plt.scatter(xr[:,0], xr[:,1], c='r', marker='+')
plt.scatter(xb[:,0], xr[:,1], c='b', marker='o')
plt.show()
