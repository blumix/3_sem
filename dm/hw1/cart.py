import numpy as np
from numba import jit
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file


@jit
def _find_best_split_feature(argsort_x, summ_all, sq_summ_all, num_all, feature_num, feature_val, min_error, num,
                             sorted_y, tmp, values_list,
                             x):
    l_size, l_sq_sum, l_sum, r_size, r_sq_sum, r_sum = _init_sums(sorted_y, values_list, summ_all, sq_summ_all, num_all)
    for val_num in values_list:
        if sorted_y[val_num] == sorted_y[val_num - 1]:
            l_size, l_sq_sum, l_sum, r_size, r_sq_sum, r_sum = \
                _change_sums(l_size, l_sq_sum, l_sum, r_size, r_sq_sum, r_sum, sorted_y[val_num])
            continue
        err = _get_mse(r_sum, r_sq_sum, r_size) + _get_mse(l_sum, l_sq_sum, l_size)

        l_size, l_sq_sum, l_sum, r_size, r_sq_sum, r_sum = \
            _change_sums(l_size, l_sq_sum, l_sum, r_size, r_sq_sum, r_sum, sorted_y[val_num])
        if not min_error or min_error > err:
            feature_num = num
            feature_val = (x[argsort_x[val_num - 1, num], num] + x[argsort_x[val_num, num], num]) / 2.
            tmp = val_num
            min_error = err

    return feature_num, feature_val, min_error, tmp


@jit
def _init_sums(sorted_y, values_list, summ_all, sq_summ_all, num_all):
    r_size = sorted_y[:values_list[0]].size
    l_size = num_all - r_size
    r_sum = sorted_y[:values_list[0]].sum()
    l_sum = summ_all - r_sum
    r_sq_sum = (sorted_y[:values_list[0]] ** 2).sum()
    l_sq_sum = sq_summ_all - r_sq_sum
    return l_size, l_sq_sum, l_sum, r_size, r_sq_sum, r_sum


@jit
def _change_sums(l_size, l_sq_sum, l_sum, r_size, r_sq_sum, r_sum, changing_val):
    r_sum += changing_val
    l_sum -= changing_val
    r_sq_sum += changing_val * changing_val
    l_sq_sum -= changing_val * changing_val
    r_size += 1
    l_size -= 1
    return l_size, l_sq_sum, l_sum, r_size, r_sq_sum, r_sum


@jit
def _get_mse(el_sum, sq_sum, num):
    return sq_sum - (el_sum * el_sum) / float(num)


class MyDecisionTreeRegressor:
    NON_LEAF_TYPE = 0
    LEAF_TYPE = 1

    def __init__(self, min_samples_split=1, max_depth=None, min_impurity_decrease=0.):
        self.tree = dict()
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease

    def __find_threshold(self, x, y):
        argsort_x = x.argsort(axis=0)

        min_error = None
        feature_num = None
        feature_val = None
        tmp = None

        values_list = range(self.min_samples_split, x.shape[0] - self.min_samples_split)
        if len(values_list) == 0:
            return None

        summ_all = np.sum(y)
        sq_summ_all = np.sum(y ** 2)
        all_num = len(y)

        all_err = _get_mse(summ_all, sq_summ_all, all_num)

        for num in range(x.shape[1]):
            sorted_y = y[argsort_x[:, num]]

            feature_num, feature_val, min_error, tmp = _find_best_split_feature(argsort_x, summ_all, sq_summ_all,
                                                                                all_num, feature_num,
                                                                                feature_val, min_error, num,
                                                                                sorted_y, tmp, values_list, x)

        if min_error is None:
            return None

        if all_err - min_error < self.min_impurity_decrease:
            return None

        return feature_num, feature_val, \
               x[argsort_x[:, feature_num]][:tmp], \
               x[argsort_x[:, feature_num]][tmp:], \
               y[argsort_x[:, feature_num]][:tmp], \
               y[argsort_x[:, feature_num]][tmp:]

    def __fit_node(self, x, y, node_id, depth):
        if self.max_depth is not None and depth == self.max_depth:
            self.tree[node_id] = (self.LEAF_TYPE, np.mean(y))
            return
        if self.min_samples_split is not None and x.shape[0] < self.min_samples_split:
            self.tree[node_id] = (self.LEAF_TYPE, np.mean(y))
            return

        res = self.__find_threshold(x, y)
        if res is None:
            self.tree[node_id] = (self.LEAF_TYPE, np.mean(y))
            return

        feature_num, feature_val, x_l, x_r, y_l, y_r = res

        self.tree[node_id] = (self.NON_LEAF_TYPE, feature_num, feature_val)
        self.__fit_node(x_l, y_l, 2 * node_id + 1, depth + 1)
        self.__fit_node(x_r, y_r, 2 * node_id + 2, depth + 1)

    def fit(self, x, y):
        self.__fit_node(x, y, 0, 0)

    def __predict(self, x, node_id):
        node = self.tree[node_id]
        if node[0] == self.__class__.NON_LEAF_TYPE:
            _, feature_num, feature_val = node
            if x[feature_num] < feature_val:
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

    def fit_predict(self, x_train, y_train, predicted_x):
        self.fit(x_train, y_train)
        return self.predict(predicted_x)


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


def test():
    X_train, X_test, y_train, y_test = load_data_2()

    err = []
    tr_err = []
    test_err = []
    tr_test_err = []

    for i in range(5, 6):
        print i
        my = MyDecisionTreeRegressor(max_depth=i)
        tree = DecisionTreeRegressor(max_depth=i)

        tree.fit(X_train, y_train)
        my.fit(X_train, y_train)

        tr_err.append(np.linalg.norm(my.predict(X_train) - y_train))
        err.append(np.linalg.norm(my.predict(X_test) - y_test))
        test_err.append(np.linalg.norm(tree.predict(X_test) - y_test))
        tr_test_err.append(np.linalg.norm(tree.predict(X_train) - y_train))

    plt.plot(tr_err, label='my_train_error')
    plt.plot(err, label='my_error')
    plt.plot(tr_test_err, label='tree_train_error')
    plt.plot(test_err, label='tree_error')
    plt.legend()
    plt.show()

# test()
