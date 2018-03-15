import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


class SplitFinder:
    def __init__(self, x, g, s, lambda_v, gamma):
        self.gamma = gamma
        self.lambda_v = lambda_v
        self.g = g
        self.x = x
        self.s = s
        self.max_gain = None
        self.feature_num = None
        self.feature_val = None
        self.splitting_number = None
        self.number_of_elements = len(self.g)
        self.sum_s = np.sum(self.s)
        self.sum_g = np.sum(self.g)
        self.values_list = range(1, self.x.shape[0])
        self.argsort_x = self.x.argsort(axis=0)

        self.Gr = None
        self.Gl = None
        self.Sr = None
        self.Sl = None

    def find_best_split(self):
        if len(self.values_list) == 0:
            return None

        for num in xrange(self.x.shape[1]):
            self.find_best_split_feature(num)

    def find_best_split_feature(self, num):
        sorted_g = self.g[self.argsort_x[:, num]]
        sorted_x = self.x[self.argsort_x[:, num], num]

        self.cur_Gr = sorted_g[:self.values_list[0]].sum()
        self.cur_Gl = sorted_g[self.values_list[0]:].sum()

        self.cur_Sr = self.values_list[0] * self.s
        self.cur_Sl = (self.number_of_elements - self.values_list[0]) * self.s
        for val_num in self.values_list:
            if sorted_x[val_num] == sorted_x[val_num - 1]:
                self.update_sum(sorted_g, val_num)
                continue

            gain = self.calculate_gain()

            self.update_sum(sorted_g, val_num)
            if not self.max_gain or self.max_gain < gain:
                self.feature_num = num
                self.feature_val = (sorted_x[val_num - 1] + sorted_x[val_num]) / 2.
                self.splitting_number = val_num
                self.max_gain = gain
                self.Gr = self.cur_Gr
                self.Gl = self.cur_Gl
                self.Sr = self.cur_Sr
                self.Sl = self.cur_Sl

    def calculate_gain(self):
        gain = self.cur_Gl * self.cur_Gl / (self.cur_Sl + self.lambda_v) + self.cur_Gr * self.cur_Gr / (
                self.cur_Sr + self.lambda_v) - (self.cur_Gl + self.cur_Gr) * (self.cur_Gl + self.cur_Gr) / (
                       self.cur_Sl + self.cur_Sr + self.lambda_v) - self.gamma
        return gain

    def update_sum(self, sorted_g, val_num):
        self.cur_Gr += sorted_g[val_num]
        self.cur_Sr += self.s

        self.cur_Gl -= sorted_g[val_num]
        self.cur_Sl -= self.s


class MyXGBoostTreeRegressor:
    NON_LEAF_TYPE = 0
    LEAF_TYPE = 1

    def __init__(self, min_samples_split=1, max_depth=None, min_impurity_decrease=0., gamma=0.1, lambda_v=0.1):
        self.lambda_v = lambda_v
        self.gamma = gamma
        self.tree = dict()
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease

    def __fit_node(self, x, g, s, node_id, depth):
        if self.max_depth is not None and depth == self.max_depth:
            self.tree[node_id] = (self.LEAF_TYPE, - g.sum() / (s * len(g) + self.lambda_v))
            return
        if self.min_samples_split is not None and x.shape[0] < self.min_samples_split:
            self.tree[node_id] = (self.LEAF_TYPE, - g.sum() / (s * len(g) + self.lambda_v))
            return

        sf = SplitFinder(x, g, s, lambda_v=self.lambda_v, gamma=self.gamma)
        sf.find_best_split()
        if sf.max_gain is None or sf.max_gain < 0:
            self.tree[node_id] = (self.LEAF_TYPE, - g.sum() / (s * len(g) + self.lambda_v))
            return

        sorted_field = sf.argsort_x[:, sf.feature_num]
        self.tree[node_id] = (self.NON_LEAF_TYPE, sf.feature_num, sf.feature_val)

        x_l = x[sorted_field][:sf.splitting_number]
        g_l = g[sorted_field][:sf.splitting_number]
        self.__fit_node(x_l, g_l, s, 2 * node_id + 1, depth + 1)

        x_r = x[sorted_field][sf.splitting_number:]
        g_r = g[sorted_field][sf.splitting_number:]
        self.__fit_node(x_r, g_r, s, 2 * node_id + 2, depth + 1)

    def fit(self, x, g, s):
        self.__fit_node(x, g, s, 0, 0)

    def __predict(self, x, node_id):
        node = self.tree[node_id]
        if node[0] == self.__class__.NON_LEAF_TYPE:
            _, feature_num, feature_val = node
            if x[feature_num] <= feature_val:
                return self.__predict(x, 2 * node_id + 1)
            else:
                return self.__predict(x, 2 * node_id + 2)
        else:
            return node[1], node_id

    def predict(self, X):
        return np.array([self.__predict(x, 0)[0] for x in X])

    def predict_node(self, X):

        ret_node = []
        ret_val = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            res = self.__predict(x, 0)
            ret_node.append(res[1])
            ret_val[i] = res[0]

        return ret_val, ret_node


class XGB:
    def __init__(self, n_estimators=10, max_depth=3, learning_rate=0.1, verbose=False, gamma=0.1, lambda_v=0.1):
        self.lambda_v = lambda_v
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.max_depth = max_depth
        self.h = None
        self.n_estimators = n_estimators
        self.trees = []
        self.tree_num = 0
        self.med = 0
        self.b = []

    def fit(self, X, y):
        self.fit_start(y)
        while self.tree_num < self.n_estimators:
            self.fit_tree(X, y)
            if self.verbose:
                print 'tree num:', self.tree_num

    def fit_start(self, y):
        self.med = np.mean(y)
        self.h = np.ones(y.shape) * self.med

    def fit_tree(self, X, y):
        g = -2 * (y - self.h)
        s = 2.
        a_i = MyXGBoostTreeRegressor(max_depth=self.max_depth, lambda_v=self.lambda_v, gamma=self.gamma)
        a_i.fit(X, g, s)
        # res = a_i.predict(X)
        b = self.learning_rate
        self.b.append(b)
        self.trees.append(a_i)
        self.h = self.h - b * a_i.predict(X)
        self.tree_num += 1

    def predict(self, X):
        res = np.ones(X.shape[0]) * self.med
        for tree in self.trees:
            res += tree.predict(X)
        return res

    def staged_predict(self, X):
        single_res = np.ones(X.shape[0]) * self.med
        ret = [np.copy(single_res)]
        for i, tree in enumerate(self.trees):
            single_res += tree.predict(X) * self.learning_rate * self.b[i]
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

    est_num = 40
    X_train, X_test, y_train, y_test = load_data_1()
    l1_err_train = []
    l1_err_test = []
    l1_test_err_train = []
    l1_test_err_test = []

    boo = XGB(n_estimators=est_num, verbose=True, learning_rate=0.1, max_depth=40, lambda_v=0., gamma=0.)
    boo.fit(X_train, y_train)
    my_res_train = boo.staged_predict(X_train)
    my_res_test = boo.staged_predict(X_test)

    test_boo = GradientBoostingRegressor(n_estimators=est_num, loss='lad', criterion='mse')
    test_boo.fit(X_train, y_train)
    res_train = list(test_boo.staged_predict(X_train))
    res_test = list(test_boo.staged_predict(X_test))

    for i in range(est_num):
        l1_err_train.append(np.linalg.norm(my_res_train[i] - y_train, ord=2))
        l1_err_test.append(np.linalg.norm(my_res_test[i] - y_test, ord=2))
        l1_test_err_train.append(np.linalg.norm(res_train[i] - y_train, ord=2))
        l1_test_err_test.append(np.linalg.norm(res_test[i] - y_test, ord=2))

    # plt.figure(figsize=(20, 10))
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
