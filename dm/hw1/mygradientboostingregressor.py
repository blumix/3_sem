import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_svmlight_file

from cart import MyDecisionTreeRegressor


class MyGradientBoostingRegressor:
    def __init__(self, n_estimators=10, max_depth=3):
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

    def fit_start(self, y):
        self.med = np.median(y)
        self.F = np.ones(y.shape) * self.med

    def fit_tree(self, X, y):
        g = np.sign(y - self.F)
        a_i = MyDecisionTreeRegressor(max_depth=self.max_depth)
        # a_i = DecisionTreeRegressor(max_depth=2)
        a_i.fit(X, g)
        _, ret_node = a_i.predict_node(X)
        # ret_node = a_i.predict(X)

        # print a_i.tree

        # print np.unique(ret_node)

        for val in np.unique(ret_node):
            index = [ret_node == val]
            gamma_m = np.median((y - self.F)[index])
            self.F[index] += gamma_m
            a_i.tree[val] = (a_i.tree[val][0], gamma_m)

        self.trees.append(a_i)
        self.tree_num += 1

    def predict(self, X):
        res = np.ones(X.shape[0]) * self.med
        for tree in self.trees:
            res += tree.predict(X)
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
    X_train, y_train = load_svmlight_file('dataset/reg.train.txt')
    X_test, y_test = load_svmlight_file('dataset/reg.test.txt')
    return X_train.toarray(), X_test.toarray(), y_train, y_test


if __name__ == '__main__':

    est_num = 200
    X_train, X_test, y_train, y_test = load_data_2()
    l1_err_train = []
    l1_err_test = []
    l1_test_err_train = []
    l1_test_err_test = []

    # l2_err_train = []
    # l2_err_test = []
    # l2_test_err_train = []
    # l2_test_err_test = []

    boo = MyGradientBoostingRegressor(n_estimators=est_num)
    boo.fit_start(y_train)

    for i in range(1, est_num):
        test_boo = GradientBoostingRegressor(n_estimators=i, loss='lad', criterion='mse', )
        boo.fit_tree(X_train, y_train)
        test_boo.fit(X_train, y_train)
        print i
        my_pred_train = boo.predict(X_train)
        my_pred_test = boo.predict(X_test)

        test_pred_train = test_boo.predict(X_train)
        test_pred_test = test_boo.predict(X_test)

        l1_err_train.append(np.linalg.norm(my_pred_train - y_train, ord=1))
        l1_err_test.append(np.linalg.norm(my_pred_test - y_test, ord=1))
        l1_test_err_train.append(np.linalg.norm(test_pred_train - y_train, ord=1))
        l1_test_err_test.append(np.linalg.norm(test_pred_test - y_test, ord=1))

        # l2_err_train.append(np.linalg.norm(my_pred_train - y_train, ord=2))
        # l2_err_test.append(np.linalg.norm(my_pred_test - y_test, ord=2))
        # l2_test_err_train.append(np.linalg.norm(test_pred_train - y_train, ord=2))
        # l2_test_err_test.append(np.linalg.norm(test_pred_test - y_test, ord=2))

    trees_num = range(0, est_num -1)
    plt.figure(figsize=(20, 10))
    # plt.subplot(121)
    plt.plot(l1_err_train, label='my train')
    plt.plot(l1_err_test, label='my test')
    plt.plot(l1_test_err_train, label='test train')
    plt.plot(l1_test_err_test, label='test test')
    plt.fill_between(trees_num, np.array(l1_test_err_test) - np.mean(l1_test_err_test) * 0.03,
                     np.array(l1_test_err_test) + np.mean(l1_test_err_test) * 0.03, alpha=0.1, color="g")
    plt.xlabel('number of estimators')
    plt.ylabel('l1 error')
    plt.legend()

    # plt.subplot(122)
    # plt.plot(l2_err_train, label='my train')
    # plt.plot(l2_err_test, label='my test')
    # plt.plot(l2_test_err_train, label='test train')
    # plt.plot(l2_test_err_test, label='test test')
    # plt.fill_between(trees_num, np.array(l2_test_err_test) - np.mean(l2_test_err_test) * 0.03,
    #                  np.array(l2_test_err_test) + np.mean(l2_test_err_test) * 0.03, alpha=0.1, color="g")
    #
    # plt.xlabel('number of estimators')
    # plt.ylabel('l2 error')
    # plt.legend()
    plt.show()
