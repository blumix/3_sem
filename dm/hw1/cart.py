import numpy as np


class MyDecisionTreeRegressor:
    NON_LEAF_TYPE = 0
    LEAF_TYPE = 1

    def __init__(self, min_samples_split=2, max_depth=None):
        self.tree = dict()
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def get_error(self, left_ids, right_ids):
        m_l = left_ids.mean ()
        m_r = right_ids.mean ()

        return np.sum(left_ids - m_l) ** 2 / left_ids.size + np.sum(right_ids - m_r) ** 2 / right_ids.size

    def __find_threshold(self, x, y):
        argsort_x = x.argsort (axis=0)

        min_error = None
        feature_num = None
        feature_val = None

        for num in range(x.shape[0]):
            sorted_y = y[argsort_x[num]]
            for val_num in range(self.min_samples_split, x.shape[1] - self.min_samples_split):
                err = self.get_error(sorted_y[:val_num], sorted_y[val_num:])
                if not min_error or min_error > err:
                    feature_num = num
                    feature_val = val_num

        return feature_num, feature_val, \
               x[argsort_x[feature_num]][:feature_val], \
               x[argsort_x[feature_num]][feature_val:], \
               y[argsort_x[feature_num]][:feature_val], \
               y[argsort_x[feature_num]][feature_val:]

    # def __div_samples (x, y, feature_num, feature_val):

    def __fit_node(self, x, y, node_id, depth, pred_f=-1):
        if self.max_depth is not None and depth == self.max_depth:
            self.tree[node_id] = (self.LEAF_TYPE, np.mean(y))
            return
        if self.min_samples_split is not None and x.shape[0] < self.min_samples_split:
            self.tree[node_id] = (self.LEAF_TYPE, np.mean(y))
            return

        feature_num, feature_val, x_l, x_r, y_l, y_r = self.__find_threshold(x, y)
        if y_l.size < self.min_samples_split or y_r.size < self.min_samples_split:
            self.tree[node_id] = (self.LEAF_TYPE, np.mean(y))
            return

        self.tree[node_id] = (self.NON_LEAF_TYPE, feature_num, feature_val)
        self.__fit_node(x_l, y_l, 2 * node_id + 1, depth + 1)
        self.__fit_node(x_r, y_r, 2 * node_id + 2, depth + 1)

    def fit(self, x, y):
        self.__fit_node(x, y, 0, 0)

    def __predict(self, x, node_id):
        node = self.tree[node_id]
        if node[0] == self.__class__.NON_LEAF_TYPE:
            _, feature_num, feature_val = node
            if x[feature_val] > feature_val:
                return self.__predict(x, 2 * node_id + 1)
            else:
                return self.__predict(x, 2 * node_id + 2)
        else:
            return node[2]

    def predict(self, X):
        return np.array([self.__predict(x, 0) for x in X])

    def fit_predict(self, x_train, y_train, predicted_x):
        self.fit(x_train, y_train)
        return self.predict(predicted_x)
