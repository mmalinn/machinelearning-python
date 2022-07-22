import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import seed
from matplotlib.colors import ListedColormap


# dane
df = pd.read_csv('iris.data', header=None)
x = df.iloc[0:100, [0, 2]].values
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
x_std = np.copy(x)


# funkcja do wizualizacji
def plot_decision_regions(x, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cl, 0], y=x[y == cl, 1], alpha=0.8,
                    marker=markers[idx], label=cl, edgecolor='black')


# liniowa funkcja aktywacji
def activation(x):
    return x

# klasyfikator - adaptacyjny liniowy neuron (Adaline SGD)


class Alone(object):
    def __init__(self, e=0.01, n_iter=10, shuffle=True, random_state=None):
        self.cost_ = None
        self.e = e
        self.n_iter = n_iter
        self.w_init = False
        self.shuffle = shuffle
        self.random_state = random_state

    # dopasowanie danych uczacych
    def fit(self, x, y):
        self._init_weights(x.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                x, y = self._shuffle(x, y)
            cost = []
            for xi, target in zip(x, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    # dopasowanie danych uczacych bez ponownej inicjalizacji wag
    def partial_fit(self, x, y):
        if not self.w_init:
            self._init_weights(x.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(x, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(x, y)
        return self

    # tasowanie danych uczacych
    def _shuffle(self, x, y):
        r = self.rgen.permutation(len(y))
        return x[r], y[r]

    # inicjacja wag
    def _init_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1+m)
        self.w_init = True

    # wykorzystanie Adaline SGD do aktualizacji wag
    def _update_weights(self, xi, target):
        output = activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.e * xi.dot(error)
        self.w_[0] += self.e * error
        cost = 0.5 * error ** 2
        return cost

    # calkowite pobudzenie
    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

    # etykieta klas po wykonaniu skoku jednostkowego
    def predict(self, x):
        return np.where(activation(self.net_input(x)) >= 0.0, 1, -1)
