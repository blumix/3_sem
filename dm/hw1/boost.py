import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from cart import MyDecisionTreeRegressor


class Boost:
    def __init__(self, number_of_trees=10, gd_iterations=1000, gd_lr=0.00005):
        self.number_of_trees = number_of_trees
        self.a = []
        self.b = []
        self.gd_iterations = gd_iterations
        self.gd_lr = gd_lr

    def gd(self, df):
        point = 1
        for i in range(self.gd_iterations):
            grad = df(point)
            point = point - self.gd_lr * grad
            if abs(self.gd_lr * grad) < 1e-10:
                break
        return point

    def fit(self, X, y):
        h = np.median(y)  # 1. init array h0
        for tree_num in range(self.number_of_trees):
            g = np.sign(y - h)
            a_i = MyDecisionTreeRegressor(max_depth=3)
            # a_i = DecisionTreeRegressor(max_depth=3)
            a_i.fit(X, -g)
            self.a.append(a_i)
            # res = a_i.predict(X)
            #             df = lambda x: np.sum (res * np.sign (y - (h + x * res)))
            #             b_i = self.gd (df)
            b_i = 1 * self.gd_lr ** (tree_num / 100.)
            self.b.append(b_i)
            h = h + b_i * h

    def predict(self, X):
        res = self.b[0] * self.a[0].predict(X)
        for tree_num in range(1, len(self.a)):
            res += self.b[tree_num] * self.a[tree_num].predict(X)
        return res


def load_data():
    all_data = pd.read_csv("auto-mpg.data",
                           delim_whitespace=True, header=None,
                           names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration',
                                  'model', 'origin', 'car_name'])
    all_data = all_data.dropna()
    y = np.array(all_data['mpg'])
    columns = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration',
               'model', 'origin']
    X = np.array(all_data[columns])
    return X, y


if __name__ == '__main__':

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    err_train = []
    err_test = []
    for i in range(1, 70):
        boo = Boost(number_of_trees=i, gd_lr=0.1, gd_iterations=15)
        boo.fit(X_train, y_train)
        if i % 50 == 0:
            print i
        err_train.append(np.linalg.norm(boo.predict(X_train) - y_train, ord=1))
        err_test.append(np.linalg.norm(boo.predict(X_test) - y_test, ord=1))

    plt.plot(err_train, c='r')
    plt.plot(err_test)
    plt.show()
