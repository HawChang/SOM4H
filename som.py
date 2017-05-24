import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import normalizer


class Som:
    def __init__(self, x, y):
        self.map = []
        self.n_neurons = x * y
        self.sigma = x
        self.template = np.arange(x * y).reshape(self.n_neurons, 1)
        self.alpha = 0.6
        self.alpha_final = 0.1
        self.shape = [x, y]
        self.epoch = 0
        self.total = None
        self.alpha_decay = None
        self.sigma_decay = None
        self.normalizer = None

    def normalize(self):
        pass

    def train(self, data, iteration=0, normalize='linear', batch_size=1):
        if iteration > data.shape[0]:
            return "iter > dataSize"
        elif iteration == 0:
            iteration = data.shape[0]
        if len(self.map) == 0:
            print("in")
            if normalize is not None:
                print("before", data, sep='\n')
                methods = {
                    'std': normalizer.Normalizer.standard_deviation_normalization,
                    'linear': normalizer.Normalizer.max_min_normalization,
                }
                data, self.normalizer = methods.get(normalize)(data)
                print("after", data, sep='\n')
            x, y = self.shape
            # map initialization
            self.map = np.zeros((self.n_neurons, len(data[0])))

            # extract the principal components of the input data
            eigen = PCA(10, svd_solver="randomized").fit_transform(data.T)
            eigen = eigen.T

            # randomize the map, better solution needed
            self.map[0] = eigen[0]
            self.map[y-1] = eigen[1]
            self.map[(x - 1) * y] = eigen[2]
            self.map[x * y - 1] = eigen[3]
            for i in range(4, min([10, len(eigen)])):
                self.map[np.random.randint(1, self.n_neurons)] = eigen[i]

        self.total = iteration

        # coefficient of decay for learning rate alpha
        self.alpha_decay = (self.alpha_final / self.alpha) ** (1.0 / self.total)

        # coefficient of decay for gaussian smoothing
        self.sigma_decay = (np.sqrt(self.shape[0]) / (4 * self.sigma)) ** (1.0 / self.total)

        # shuffle training data, for better model
        samples = np.arange(len(data))
        np.random.shuffle(samples)

        # training the model
        for i in range(iteration):
            idx = samples[i:i + batch_size]
            # print(idx)
            self.iterate(data[idx])
            # self.display()
        # plt.show()

    def transform(self, data):
        # We simply compute the dot product of the input with the transpose of the map to get the new input vectors
        res = np.dot(np.exp(data), np.exp(self.map.T)) / np.sum(np.exp(self.map), axis=1)
        res = res / (np.exp(np.max(res)) + 1e-8)
        return res

    def iterate(self, vector):
        x, y = self.shape

        # Euclidean distance of each neurons with the input data
        delta = self.map - vector
        dists = np.sum(delta ** 2, axis=1).reshape(x, y)

        # active neuron position
        idx = np.argmin(dists)
        row, col = divmod(idx, y)
        print("Epoch ", self.epoch, ": ", (row, col), "; Sigma: ", self.sigma, "; alpha: ", self.alpha)

        # Linearly reducing the width of Gaussian Kernel
        self.sigma = self.sigma * self.sigma_decay
        dist_map = self.template.reshape(x, y)

        # Distance of each neurons in the map from the best matching neuron
        dists = np.sqrt((dist_map / x - row) ** 2 + (np.mod(dist_map, x) - col) ** 2).reshape(self.n_neurons, 1)

        # dists = self.template - idx

        # Applying Gaussian smoothing to distances of neurons from best matching neuron
        h = np.exp(-(dists / self.sigma) ** 2)

        # Updating neurons in the map
        self.map -= self.alpha * h * delta

        # Decreasing alpha
        self.alpha = self.alpha * self.alpha_decay

        # record epoch
        self.epoch = self.epoch + 1

    # display the SOM model by each neuron's neighbor area size
    def display(self, position=111):
        neuron_map = self.map
        row, col = self.shape
        map_dist = []
        for i in range(row):
            for j in range(col):
                pos = i * row + j
                neighbor_size = 0
                distance = 0
                if i > 0:
                    distance += np.sqrt(np.sum((neuron_map[pos] - neuron_map[pos - row]) ** 2))
                    neighbor_size += 1
                if i < row - 1:
                    distance += np.sqrt(np.sum((neuron_map[pos] - neuron_map[pos + row]) ** 2))
                    neighbor_size += 1
                if j > 0:
                    distance += np.sqrt(np.sum((neuron_map[pos] - neuron_map[pos - 1]) ** 2))
                    neighbor_size += 1
                if j < col - 1:
                    distance += np.sqrt(np.sum((neuron_map[pos] - neuron_map[pos + 1]) ** 2))
                    neighbor_size += 1
                map_dist.append(distance / neighbor_size)
        max_dist = map_dist[np.argmax(map_dist)]
        for i in range(row):
            for j in range(col):
                plt.subplot(position).scatter(i, j, s=20 * map_dist[i * row + j] / max_dist, c='k', alpha=1)
        # plt.show()


# deprecated
def test_plot(som_map, test, position):
    delta = som_map.map - test
    row, col = som_map.shape
    print(row, col)
    # Euclidean distance of each neurons with the example
    dists = np.sum(delta ** 2, axis=1).reshape(row, col)
    # Best matching unit
    idx = np.argmin(dists)
    # print(idx)
    m, n = divmod(idx, col)
    # print(m, n)
    alphas = dists[m][n] / dists
    for i in range(row):
        for j in range(col):
            plt.subplot(position).scatter(i, j, s=100, c='b', alpha=alphas[i][j])


def cluster(som_map, test_data, target, position=111):
    # test_data = normalizor.Normalizer.standard_deviation_normalization(test_data, som_map.normalizer)
    print(len(test_data))
    for i in range(len(test_data)):
        # print(i)
        delta = som_map.map - test_data[i]
        row, col = som_map.shape
        dists = np.sum(delta ** 2, axis=1).reshape(row, col)
        idx = np.argmin(dists)
        m, n = divmod(idx, col)
        print("pos:(", m, ",", n, ")")
        # target = list(map(int, target))
        colors = {
            'Iris-setosa': 'b',    # 蓝色   'Iris-setosa'
            'Iris-versicolor': 'r',    # 红色
            'Iris-virginica': 'y',    # 黄色
            # 0: 'b',  # 蓝色
            # 2: 'r',  # 红色
            # 1: 'y',  # 黄色
        }
        row_offset = np.random.rand()*1.5-0.75
        # print(dists[m][n])
        col_offset = np.random.rand() * 1.5 - 0.75
        # col_offset = np.sqrt(np.power(dists[m][n], 2) - np.power(row_offset, 2))
        # print(col_offset)
        plt.subplot(position).scatter(float(m)+row_offset, float(n)+col_offset, s=20,
                                      c=colors.get(target[i]), alpha=1)
    # plt.show()


def read_csv_file(address, header=True, sep=','):
    data = []
    label = []
    attributes_name = []
    with open(address) as csv_file:
        line = csv_file.readline()
        if line and not attributes_name:
            label = [True for i in range(len(line.split(sep)))]
            if header is True:
                attributes_name = [item.strip('\n ') for item in line.split(sep)]
                line = csv_file.readline()
            else:
                num = len(line.split(sep))
                for i in range(num):
                    attributes_name.append('V'+str(i))
        while line:
            record = [item.strip('\t\n ') for item in line.split(sep)]
            for i in range(len(label)):
                if label[i]:
                    if is_float(record[i]):
                        continue
                    else:
                        label[i] = False
            # record = [item.strip('\n ') for item in line.split(sep)]
            #
            # record = [item.strip('\n ') for item in line.split(sep)]
            # label.append(record[-1])
            # record = [i for i in map(float, record[:-1])]
            data.append(record)
            line = csv_file.readline()
    # print(label)
    # data = np.array(data)
    # formats = []
    # for i in range(len(label)):
    #     if not label[i]:
    #         # formats.append('f')
    #         for x in data:
    #             # x[i] = float(x[i])
    #             del x[i]
        # else:
        #    formats.append('S32')
    # data_type = np.dtype({
    #     'names': attributes,
    #     'formats': formats
    # })
    # print(attributes)
    # print(formats)
    # data = np.array(data, dtype=data_type)

    # print(data)
    return np.array(data), attributes_name


def is_float(data):
    try:
        float(data)
        return True
    except:
        return False


def demo():
    # data, attributes = read_csv_file('train.csv')
    # test, trash = read_csv_file('train.csv', False)
    # train = data[:, :-1]
    # test = data[:, :-1]
    # print(train[:, 0:4])
    # train = np.array(train[:, :-1])
    # print(train)
    # test, trash = read_csv_file('iris.csv', False)
    # test = train[:, 4]
    # train = train[:, 0:4]
    # train = train[:, 0:4]
    # print(train)
    # print(test)
    # print(data[0])
    # print(attrs)
    # print(train.shape)
    # train = train[:, 0:206]
    # print(train[0])
    # print(test)

    # log测试
    # data, attributes = read_csv_file('train.csv')
    # test, trash = read_csv_file('test.csv', False)
    # train = np.array(data[:, :-1], dtype=np.float)
    # test = np.array(test[:, :], dtype=np.float)
    # target = test[:, -1]
    # som_map = Som(20, 20)
    # print(train)
    # som_map.train(train)
    # test = normalizer.Normalizer.standard_deviation_normalization(test[:, :-1], som_map.normalizer)
    # print(test)
    # cluster(som_map, test, target)

    # iris测试
    data, attributes = read_csv_file('iris.csv')
    train = np.array(data[:, :-1], dtype=np.float)
    target = data[:, -1]
    som_map = Som(10, 10)
    som_map.train(train, normalize='std')
    cluster(som_map, train, target)


    #
    #
    #
    som_map.display()
    plt.show()

if __name__ == '__main__':
    demo()
