import numpy as np
from sklearn.decomposition import PCA


class som():
    def __init__(self, x, y):
        self.map = []
        self.n_neurons = x * y
        self.sigma = x
        self.template = np.arange(x * y).reshape(self.n_neurons, 1)
        self.alpha = 0.6
        self.alpha_final = 0.1
        self.shape = [x, y]
        self.epoch = 0

    def train(self, data, iteration=0, batch_size=1):
        if  iteration> data.shape[0]:
            return "iter > dataSize"
        elif iteration == 0:
            iteration = data.shape[0]
        if len(self.map) == 0:
            x, y = self.shape
            # first we initialize the map
            self.map = np.zeros((self.n_neurons, len(data[0])))

            # then we the pricipal components of the input data
            eigen = PCA(10, svd_solver="randomized").fit_transform(data.T)
            print(data.T.shape)
            print(eigen.shape)
            print(self.map.shape)
            eigen = eigen.T
            # then we set different point on the map equal to principal components to force diversification
            self.map[0] = eigen[0]
            self.map[y-1] = eigen[1]
            self.map[(x - 1) * y] = eigen[2]
            self.map[x * y - 1] = eigen[3]
            # for i in range(4, 10):
            #     self.map[numpy.random.randint(1, self.n_neurons)] = eigen[i]

        self.total = iteration

        # coefficient of decay for learning rate alpha
        self.alpha_decay = (self.alpha_final / self.alpha) ** (1.0 / self.total)

        # coefficient of decay for gaussian smoothing
        self.sigma_decay = (np.sqrt(self.shape[0]) / (4 * self.sigma)) ** (1.0 / self.total)

        samples = np.arange(len(data))
        np.random.shuffle(samples)

        for i in range(iteration):
            idx = samples[i:i + batch_size]
            print(idx)
            self.iterate(data[idx])

    def transform(self, X):
        # We simply compute the dot product of the input with the transpose of the map to get the new input vectors
        res = np.dot(np.exp(X), np.exp(self.map.T)) / np.sum(np.exp(self.map), axis=1)
        res = res / (np.exp(np.max(res)) + 1e-8)
        return res

    def iterate(self, vector):
        x, y = self.shape
        # print('map:\n', self.map)
        # print('vector:\n', vector)
        delta = self.map - vector

        # Euclidian distance of each neurons with the example
        dists = np.sum((delta) ** 2, axis=1).reshape(x, y)

        # Best maching unit
        idx = np.argmin(dists)
        print("Epoch ", self.epoch, ": ", (int(idx / x), idx % y), "; Sigma: ", self.sigma, "; alpha: ", self.alpha)

        # Linearly reducing the width of Gaussian Kernel
        self.sigma = self.sigma * self.sigma_decay
        dist_map = self.template.reshape(x, y)

        # Distance of each neurons in the map from the best matching neuron
        dists = np.sqrt((dist_map / x - idx / x) ** 2 + (np.mod(dist_map, x) - idx % y) ** 2).reshape(
            self.n_neurons, 1)
        # dists = self.template - idx

        # Applying Gaussian smoothing to distances of neurons from best matching neuron
        h = np.exp(-(dists / self.sigma) ** 2)

                # Updating neurons in the map
        self.map -= self.alpha * h * delta

        # Decreasing alpha
        self.alpha = self.alpha * self.alpha_decay

        self.epoch = self.epoch + 1


import csv
import matplotlib.pyplot as plt


def testPlot(somMap, test, position):
    delta = somMap.map - test
    row, col = somMap.shape
    print(row,col)
    # Euclidian distance of each neurons with the example
    dists = np.sum(delta ** 2, axis=1).reshape(row, col)
    # Best maching unit
    idx = np.argmin(dists)
    # print(idx)
    m, n = divmod(idx, col)
    # print(m, n)
    alphas = dists[m][n] / dists
    for i in range(row):
        for j in range(col):
            plt.subplot(position).scatter(i, j, s=10, c='b', alpha=alphas[i][j])


def demo():
    train = []
    test = []
    with open('iris.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # sepal.length,sepal.width,petal.length,petal.width,Species
            train.append(list(map(float, [row['sepal.length'], row['sepal.width'], row['petal.length'], row['petal.width']])))
            test.append(row['Species '])
    train = np.array(train)
    somMap = som(20, 20)
    # print(train.shape)
    somMap.train(train)

    testPlot(somMap, train[10], 221)
    testPlot(somMap, train[40], 222)
    testPlot(somMap, train[80], 223)
    testPlot(somMap, train[120], 224)
    plt.show()

if __name__ == '__main__':
    demo()
