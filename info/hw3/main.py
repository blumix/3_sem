import pandas as pd
import numpy as np

import LambdaMART

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import matplotlib.pyplot as plt


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
        predictor.fit()
        scores = predictor.predict(test.as_matrix(columns=test_columns))

        error = 0
        # min_id = test[0]
        for qid in test_qid:
            ind = test[test.QID == qid].index
            ndcg_my = LambdaMART.dcg_k(scores[ind], 5) / LambdaMART.ideal_dcg_k(scores[ind], 5)
            ndcg_res = LambdaMART.dcg_k(test[test.QID == qid].Y.as_matrix(), 5) / LambdaMART.ideal_dcg_k(
                test[test.QID == qid].Y.as_matrix(), 5)
            error += (ndcg_my - ndcg_res) ** 2

        error_res.append(error)

    plt.plot(error_res)
    plt.show()


def main():
    train = pd.read_csv("data/train.data.cvs")
    not_ranked = [qid for qid in set(train.QID) if len(set(train.Y[train.QID == qid])) == 1]
    for qid in not_ranked:
        train = train.drop(train[train.QID == qid].index, axis=0)
    train = train.reset_index(drop=True)

    logging.info('Useless data dropped.')

    train['Y'] = train['Y'].apply(np.log)
    X_columns = [col for col in train.columns if col[0] == u'X']
    train_colums = ['Y', 'QID'] + X_columns
    train_X = train.as_matrix(columns=train_colums)

    predictor = LambdaMART.LambdaMART(training_data=train_X, number_of_trees=150)

    error_res = []
    for scores in predictor.fit():
        error = 0
        for qid in set(train.QID):
            ind = train[train.QID == qid].index
            ndcg_my = LambdaMART.dcg_k(scores[ind], 5) / LambdaMART.ideal_dcg_k(scores[ind], 5)
            ndcg_res = LambdaMART.dcg_k(train[train.QID == qid].Y.as_matrix(), 5) / LambdaMART.ideal_dcg_k(
                train[train.QID == qid].Y.as_matrix(), 5)
            error += (ndcg_my - ndcg_res) ** 2

        error_res.append(np.log(error))
        plt.plot(error_res)
        plt.show()
        print(error)

    predictor.save("temp/mart.dump")
    test = pd.read_csv("data/testset.cvs")
    test_columns = ['QID'] + X_columns

    scores = predictor.predict(test.as_matrix(columns=test_columns))
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
            f.write("{}, {}\n".format(ind[sd[i]], q))


if __name__ == '__main__':
    run_num = int(open("run_number", "r").readline())
    run_num += 1
    open("run_number", "w").write("{}".format(run_num))
    print("Run number: ", run_num)

    main()
