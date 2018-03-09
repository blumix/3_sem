import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_svmlight_file

from cart import MyDecisionTreeRegressor


class MyGradientBoostingRegressor:
    def __init__(self, n_estimators=10, max_depth=3, learning_rate=0.1, verbose=False):
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.max_depth = max_depth
        self.F = None
        self.n_estimators = n_estimators
        self.trees = []
        self.tree_num = 0
        self.med = 0

    def fit(self, X, y):
        self.fit_start(y)
        while self.tree_num < self.n_estimators:
            self.fit_tree(X, y)
            if self.verbose:
                print 'tree num:', self.tree_num

    def fit_start(self, y):
        self.med = np.median(y)
        self.F = np.ones(y.shape) * self.med

    def fit_tree(self, X, y):
        g = np.sign(y - self.F)
        a_i = MyDecisionTreeRegressor(max_depth=self.max_depth)
        a_i.fit(X, g)
        _, ret_node = a_i.predict_node(X)

        for val in np.unique(ret_node):
            index = [ret_node == val]
            gamma_m = np.median((y - self.F)[index])
            self.F[index] += gamma_m * self.learning_rate
            a_i.tree[val] = (a_i.tree[val][0], gamma_m)

        self.trees.append(a_i)
        self.tree_num += 1

    def predict(self, X):
        res = np.ones(X.shape[0]) * self.med
        for tree in self.trees:
            res += tree.predict(X)
        return res

    def staged_predict(self, X):
        single_res = np.ones(X.shape[0]) * self.med
        ret = [np.copy(single_res)]
        for tree in self.trees:
            single_res += tree.predict(X) * self.learning_rate
            temp = np.copy(single_res)
            ret.append(temp)
        return ret


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
    X_train, y_train = load_svmlight_file('dataset/reg.train.txt')
    X_test, y_test = load_svmlight_file('dataset/reg.test.txt')
    return X_train.toarray(), X_test.toarray(), y_train, y_test


if __name__ == '__main__':

    est_num = 600
    X_train, X_test, y_train, y_test = load_data_2()
    l1_err_train = []
    l1_err_test = []
    l1_test_err_train = []
    l1_test_err_test = []

    boo = MyGradientBoostingRegressor(n_estimators=est_num, verbose=True, learning_rate=0.1)
    boo.fit(X_train, y_train)
    my_res_train = boo.staged_predict(X_train)
    my_res_test = boo.staged_predict(X_test)

    test_boo = GradientBoostingRegressor(n_estimators=est_num, loss='lad', criterion='mse')
    test_boo.fit(X_train, y_train)
    res_train = list(test_boo.staged_predict(X_train))
    res_test = list(test_boo.staged_predict(X_test))

    for i in range(est_num):
        l1_err_train.append(np.linalg.norm(my_res_train[i] - y_train, ord=1))
        l1_err_test.append(np.linalg.norm(my_res_test[i] - y_test, ord=1))
        l1_test_err_train.append(np.linalg.norm(res_train[i] - y_train, ord=1))
        l1_test_err_test.append(np.linalg.norm(res_test[i] - y_test, ord=1))

    plt.figure(figsize=(20, 10))
    plt.plot(l1_err_train, label='my train')
    plt.plot(l1_err_test, label='my test')
    plt.plot(l1_test_err_train, label='test train')
    plt.plot(l1_test_err_test, label='test test')
    plt.fill_between(range(est_num), np.array(l1_test_err_test) - np.mean(l1_test_err_test) * 0.03,
                     np.array(l1_test_err_test) + np.mean(l1_test_err_test) * 0.03, alpha=0.1, color="g")
    plt.fill_between(range(est_num), np.array(l1_test_err_train) - np.mean(l1_test_err_train) * 0.03,
                     np.array(l1_test_err_train) + np.mean(l1_test_err_train) * 0.03, alpha=0.1, color="g")
    plt.xlabel('number of estimators')
    plt.ylabel('l1 error')
    plt.legend()
    plt.show()
