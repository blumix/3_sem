import pandas as pd
import numpy as np

import LambdaMART

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import matplotlib.pyplot as plt

from sklearn.datasets import dump_svmlight_file

import xgboost


def test():
    train = pd.read_csv("data/train.data.cvs")
    not_ranked = [qid for qid in set(train.QID) if len(set(train.Y[train.QID == qid])) == 1]

    test = pd.DataFrame(train[-5000:]).reset_index()
    train = pd.DataFrame(train[:-5000])

    for qid in not_ranked:
        train = train.drop(train[train.QID == qid].index, axis=0)

    logging.info('Useless data dropped.')

    train['Y'] = train['Y'].apply(np.log)
    test['Y'] = test['Y'].apply(np.log)

    X_columns = [col for col in train.columns if col[0] == u'X']
    train_colums = ['Y', 'QID'] + X_columns
    train_X = train.as_matrix(columns=train_colums)
    test_columns = ['QID'] + X_columns
    test_qid = set(test.QID)

    error_res = []

    for tre_num in range(1, 15):
        predictor = LambdaMART.LambdaMART(training_data=train_X, number_of_trees=tre_num)
        predictor.fit(None)
        scores = predictor.predict(test.as_matrix(columns=test_columns))

        error = []
        # min_id = test[0]
        for qid in test_qid:
            ind = test[test.QID == qid].index
            # ndcg_my = LambdaMART.ideal_dcg_k(scores[ind], 5) / LambdaMART.ideal_dcg_k(scores[ind], 5)
            my_sort = np.argsort(-scores[ind])
            ndcg_res = LambdaMART.dcg_k(test[test.QID == qid].Y.as_matrix()[my_sort], 5) / LambdaMART.ideal_dcg_k(
                test[test.QID == qid].Y.as_matrix(), 5)
            error.append(ndcg_res)

        error_res.append(np.mean(error))

        plt.plot(error_res)
        plt.show()

    plt.plot(error_res)
    plt.show()


def main():
    train = pd.read_csv("data/train.data.cvs")  # , nrows=10000)
    logging.info("data was read.")
    not_ranked = [qid for qid in set(train.QID) if len(set(train.Y[train.QID == qid])) == 1]
    for qid in not_ranked:
        train = train.drop(train[train.QID == qid].index, axis=0)
    train = train.reset_index(drop=True)

    logging.info('Useless data dropped.')

    train['Y'] = train['Y'].apply(np.log)
    X_columns = [col for col in train.columns if col[0] == u'X']
    train_colums = ['Y', 'QID'] + X_columns

    train_set = np.random.choice(list(set(train.QID)), int(len(set(train.QID))), replace=False)
    test_set = [qid for qid in set(train.QID) if qid not in train_set]

    test = train[train.QID.isin(test_set)]
    train = train[train.QID.isin(train_set)]
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    test_columns = ['QID'] + X_columns
    predictor = LambdaMART.LambdaMART(training_data=train.as_matrix(columns=train_colums), number_of_trees=2000, learning_rate=1.0)

    ndcg_train = []
    ndcg_test = []
    for scores_train, scores_test in predictor.fit(test_data=test.as_matrix(columns=test_columns)):
        n_train = []
        for qid in set(train.QID):
            ind = train[train.QID == qid].index

            my_sort = np.argsort(-scores_train[ind])
            ndcg_res = LambdaMART.dcg_k(train[train.QID == qid].Y.as_matrix()[my_sort], 5) / LambdaMART.ideal_dcg_k(
                train[train.QID == qid].Y.as_matrix(), 5)
            # ndcg_my = LambdaMART.dcg_k(scores_train[ind], 5) / LambdaMART.ideal_dcg_k(scores_train[ind], 5)
            # ndcg_res = LambdaMART.dcg_k(train[train.QID == qid].Y.as_matrix(), 5) / LambdaMART.ideal_dcg_k(
            #     train[train.QID == qid].Y.as_matrix(), 5)
            n_train.append(ndcg_res)
            # error_train += (ndcg_my - ndcg_res) ** 2

        # n_test = []
        # for qid in set(test.QID):
        #     ind = test[test.QID == qid].index
        #
        #     my_sort = np.argsort(-scores_test[ind])
        #     ndcg_res = LambdaMART.dcg_k(test[test.QID == qid].Y.as_matrix()[my_sort], 5) / LambdaMART.ideal_dcg_k(
        #         test[test.QID == qid].Y.as_matrix(), 5)
        #     # ndcg_my = LambdaMART.dcg_k(scores_test[ind], 5) / LambdaMART.ideal_dcg_k(scores_test[ind], 5)
        #     # ndcg_res = LambdaMART.dcg_k(test[test.QID == qid].Y.as_matrix(), 5) / LambdaMART.ideal_dcg_k(
        #     #     test[test.QID == qid].Y.as_matrix(), 5)
        #     n_test.append(ndcg_res)
        #     # error_test += (ndcg_my - ndcg_res) ** 2

        ndcg_train.append(np.mean(n_train))
        # ndcg_test.append(np.mean(n_test))
        plt.plot(ndcg_train, label='train')
        # plt.plot(ndcg_test, label='test')
        plt.legend()
        plt.show()
        # print(error_train, error_test)
        print(ndcg_train[-1])

    predictor.save("temp/mart.dump")
    get_answer()


def get_answer():
    predictor = LambdaMART.LambdaMART()
    predictor.load("temp/temp_model_420_lr01.lmart")

    test = pd.read_csv("data/testset.cvs")
    X_columns = [col for col in test.columns if col[0] == u'X']

    test_columns = ['QID'] + X_columns
    scores = predictor.predict(test.as_matrix(columns=test_columns), num_of_trees=5)
    qid = set(test.QID)
    global run_num
    f = open("results/try_{}.csv".format(run_num), "w")
    f.write("DocumentId,QueryId\n")
    for q in sorted(qid):
        ind = test[test.QID == q].index
        sd = np.argsort(-scores[ind])
        lem = len(sd)
        if lem > 5:
            lem = 5
        for i in range(lem):
            print(f"Qid:{q}, {ind[sd[i]]}, score:{scores[ind][sd[i]]}")
            f.write("{},{}\n".format(ind[sd[i]], q))


def counstruct_data_for_xgb():
    train = pd.read_csv("data/train.data.cvs")  # , nrows=10000)
    logging.info("data was read.")
    not_ranked = [qid for qid in set(train.QID) if len(set(train.Y[train.QID == qid])) == 1]
    for qid in not_ranked:
        train = train.drop(train[train.QID == qid].index, axis=0)
    train = train.reset_index(drop=True)

    logging.info('Useless data dropped.')

    # train['Y'] = train['Y'].apply(np.log)

    X_columns = [col for col in train.columns if col[0] == u'X']
    dump_svmlight_file(train.as_matrix(columns=X_columns), y=train.as_matrix(columns=['Y']).ravel(),
                       f="temp/xgboost_train_data")

    f = open("temp/xgboost_train_data.group", "w")
    cur_len = 0
    prev_qid = train.QID[0]
    for qid in train.as_matrix(columns=['QID']).ravel():
        if qid != prev_qid and cur_len > 0:
            f.write(f"{cur_len}\n")
            cur_len = 1
            prev_qid = qid
        else:
            cur_len += 1
    f.close()

    train = pd.read_csv("data/testset.cvs")  # , nrows=10000)
    logging.info("data was read.")

    X_columns = [col for col in train.columns if col[0] == u'X']
    dump_svmlight_file(train.as_matrix(columns=X_columns), y=np.zeros(len(train)),
                       f="temp/xgboost_test_data")

    f = open("temp/xgboost_test_data.group", "w")
    cur_len = 0
    prev_qid = train.QID[0]
    for qid in train.as_matrix(columns=['QID']).ravel():
        if qid != prev_qid and cur_len > 0:
            f.write(f"{cur_len}\n")
            cur_len = 1
            prev_qid = qid
        else:
            cur_len += 1
    f.close()


def run_xgboost():
    train = xgboost.DMatrix("temp/xgboost_train_data")
    params = {'objective': "rank:pairwise", 'max_depth': 6, 'num_round': 50, 'save_period': 5, 'nthread': 8, 'n_jobs': 8, 'n_estimators':200}

    booster = xgboost.train(params=params, dtrain=train, num_boost_round=1000)
    test = xgboost.DMatrix("temp/xgboost_test_data")
    scores = booster.predict(test)

    test = pd.read_csv("data/testset.cvs")
    qid = set(test.QID)
    global run_num
    f = open("results/try_{}.csv".format(run_num), "w")
    f.write("DocumentId,QueryId\n")
    for q in sorted(qid):
        ind = test[test.QID == q].index
        sd = np.argsort(-scores[ind])
        lem = len(sd)
        if lem > 5:
            lem = 5
        for i in range(lem):
            print(f"Qid:{q}, {ind[sd[i]]}, score:{scores[ind][sd[i]]}")
            f.write("{},{}\n".format(ind[sd[i]], q))


if __name__ == '__main__':
    run_num = int(open("run_number", "r").readline())
    run_num += 1
    open("run_number", "w").write("{}".format(run_num))
    print("Run number: ", run_num)
    # test()
    main()
    # get_answer()
    # counstruct_data_for_xgb()

    # run_xgboost()