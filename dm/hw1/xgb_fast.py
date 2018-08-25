import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

from numba import jit

@jit
def find_best_split(x, g, s, lambda_v, gamma):
    max_gain = None
    feature_num = None
    feature_val = None
    splitting_number = None
    number_of_elements = len(g)
    values_list = range(1, x.shape[0])
    argsort_x = x.argsort(axis=0)

    Gr = None
    Gl = None
    Sr = None
    Sl = None

    l_gain = None
    r_gain = None

    if len(values_list) == 0:
        return None

    for num in xrange(x.shape[1]):
        sorted_g = g[argsort_x[:, num]]
        sorted_x = x[argsort_x[:, num], num]

        cur_Gr = sorted_g[:values_list[0]].sum()
        cur_Gl = sorted_g[values_list[0]:].sum()

        cur_Sr = values_list[0] * s
        cur_Sl = (number_of_elements - values_list[0]) * s

        cur_gain_all = (cur_Gl + cur_Gr) * (cur_Gl + cur_Gr) / (
                cur_Sl + cur_Sr + lambda_v)

        for val_num in values_list:
            if sorted_x[val_num] == sorted_x[val_num - 1]:
                cur_Gr += sorted_g[val_num]
                cur_Sr += s

                cur_Gl -= sorted_g[val_num]
                cur_Sl -= s
                continue

            l_gain_cur = cur_Gl * cur_Gl / (cur_Sl + lambda_v)
            r_gain_cur = cur_Gr * cur_Gr / (cur_Sr + lambda_v)
            gain = l_gain_cur + r_gain_cur - cur_gain_all - gamma

            cur_Gr += sorted_g[val_num]
            cur_Sr += s

            cur_Gl -= sorted_g[val_num]
            cur_Sl -= s

            if not max_gain or max_gain < gain:
                feature_num = num
                feature_val = (sorted_x[val_num - 1] + sorted_x[val_num]) / 2.
                splitting_number = val_num
                max_gain = gain
                l_gain = l_gain_cur
                r_gain = r_gain_cur

                Gr = cur_Gr
                Gl = cur_Gl
                Sr = cur_Sr
                Sl = cur_Sl
    return feature_num, feature_val, splitting_number, l_gain, r_gain, Gr, Gl, Sr, Sl, max

