import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_svmlight_file

from cart import MyDecisionTreeRegressor


class MyGradientBoostingRegressor:
    def __init__(self, n_estimators=10):
        self.h = None
        self.n_estimators = n_estimators
        self.a = []
        self.b = []
        self.tree_num = 0
        self.med = 0

    def fit(self, X, y):
        self.fit_start(y)
        while self.tree_num < self.n_estimators:
            self.fit_tree(X, y)

    def fit_start(self, y):
        self.med = np.median(y)
        self.h = np.ones(y.shape) * self.med

    def fit_tree(self, X, y):
        g = np.sign(y - self.h)
        # a_i = MyDecisionTreeRegressor(max_depth=4)
        a_i = DecisionTreeRegressor(max_depth=5)
        a_i.fit(X, g)
        self.a.append(a_i)
        # _, ret_node = a_i.predict_node(X)
        ret_node = a_i.predict(X)

        print np.unique(ret_node)

        for val in np.unique(ret_node):
            index = [ret_node == val]
            # print index
            median = np.median((y - self.h)[index])
            # print median
            self.h[index] += median

        # sorted = np.argsort(np.abs(res_val))
        # all_sum = np.sum(np.abs(res_val))
        # sum_l = 1
        # index = 1
        #
        # sum_r = res_val[sorted[0]] / all_sum
        # sum_l -= res_val[sorted[0]] / all_sum
        #
        # len_y = len(sorted)
        #
        # while (sum_l > 1. / 2 or sum_r > 1. / 2) and index < len_y:
        #     print sum_l, sum_r
        #     sum_r += abs(res_val[sorted[index]]) / all_sum
        #     sum_l -= abs(res_val[sorted[index]]) / all_sum
        #     index += 1
        #
        # real_index = sorted[index]
        b_i = 1
        self.b.append(b_i)
        # self.h = self.h + b_i * res_val
        self.tree_num += 1

    def predict(self, X):
        res = np.ones(X.shape[0]) * self.med
        for tree_num in range(len(self.a)):
            res += self.a[tree_num].predict(X)
        return res


def load_data_1():
    all_data = pd.read_csv("auto-mpg.data",
                           delim_whitespace=True, header=None,
                           names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration',
                                  'model', 'origin', 'car_name'])
    all_data = all_data.dropna()
    y = np.array(all_data['mpg'])
    columns = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration',
               'model', 'origin']
    X = np.array(all_data[columns])
    return train_test_split(X, y, test_size=0.33, random_state=42)


def load_data_2():
    X_train, y_train = load_svmlight_file('Regression dataset/reg.train.txt')
    X_test, y_test = load_svmlight_file('Regression dataset/reg.test.txt')
    return X_train.toarray(), X_test.toarray(), y_train, y_test


if __name__ == '__main__':

    X_train, X_test, y_train, y_test = load_data_1()

    err_train = []
    err_test = []

    test_err_train = []
    test_err_test = []

    boo = MyGradientBoostingRegressor(n_estimators=30)
    boo.fit_start(y_train)

    for i in range(1, 105):
        test_boo = GradientBoostingRegressor(n_estimators=i, loss='lad', criterion='mse')
        boo.fit_tree(X_train, y_train)
        test_boo.fit(X_train, y_train)
        print i
        err_train.append(np.linalg.norm(boo.predict(X_train) - y_train, ord=1))
        err_test.append(np.linalg.norm(boo.predict(X_test) - y_test, ord=1))

        test_err_train.append(np.linalg.norm(test_boo.predict(X_train) - y_train, ord=1))
        test_err_test.append(np.linalg.norm(test_boo.predict(X_test) - y_test, ord=1))

    plt.plot(err_train, label='my_train')
    plt.plot(err_test, label='my_test')

    plt.plot(test_err_train, label='test_train')
    plt.plot(test_err_test, label='test_test')

    plt.legend()
    plt.show()
