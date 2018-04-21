import pandas as pd
import numpy as np
import LambdaMART
import logging
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def prepare_data():
    train = pd.read_csv("data/train.data.cvs")  # , nrows=20000)
    logging.info("data was read.")
    not_ranked = [qid for qid in set(train.QID) if len(set(train.Y[train.QID == qid])) == 1]
    for qid in not_ranked:
        train = train.drop(train[train.QID == qid].index, axis=0)
    train = train.reset_index(drop=True)

    for qid in set(train.QID):
        for num, val, in enumerate(sorted(set(train.Y[train.QID == qid]))):
            train.loc[(train.QID == qid) & (train.Y == val), 'Y'] = num + 1

    train = train.sort_values(by=['QID', 'Y'], ascending=[True, False])
    train = train.reset_index(drop=True)
    logging.info(f"{len (train)} prepared.")
    train.to_csv("data/prepared_train.csv")


def test():
    train = pd.read_csv("data/prepared_train.csv")  # , nrows=20000)
    logging.info("data was read.")
    X_columns = [col for col in train.columns if col[0] == u'X']
    predictor = LambdaMART.LambdaMART()
    predictor.load("temp/temp_model_630.lmart")
    test_columns = ['QID'] + X_columns
    scores = predictor.predict(train.as_matrix(columns=test_columns))

    n_train = []
    for qid in set(train.QID):
        ind = train[train.QID == qid].index

        my_sort = np.argsort(scores[ind])
        ndcg_res = LambdaMART.dcg(train[train.QID == qid].Y.as_matrix()[my_sort]) / LambdaMART.ideal_dcg(
            train[train.QID == qid].Y.as_matrix())
        n_train.append(ndcg_res)

    print(np.mean(n_train))


def main():
    train = pd.read_csv("data/prepared_train.csv")  # , nrows=100)
    logging.info("data was read.")
    # inds = []
    # for qid in sorted(set(train.QID), reverse=True):
    #     prev_y = -1
    #     for y_i in train[train.QID == qid].index:
    #         if train.Y[y_i] != prev_y:
    #             prev_y = train.Y[y_i]
    #             inds.append(y_i)
    # # train = train.loc[inds[:2]]
    # train = train.reset_index(drop=True)

    X_columns = [col for col in train.columns if col[0] == u'X']

    X = train.as_matrix(columns=X_columns)
    y = train.Y.as_matrix()
    qids = train.QID.as_matrix()

    predictor = LambdaMART.LambdaMART(number_of_trees=1500, learning_rate=0.1, max_depth=4)

    ndcg_train = []
    for scores_train in predictor.fit(X=X, y=y, qid=qids):
        n_train = []
        for qid in set(train.QID):
            ind = train[train.QID == qid].index

            my_sort = np.argsort(scores_train[ind])[::-1]
            ndcg_res = LambdaMART.dcg(train[train.QID == qid].Y.as_matrix()[my_sort]) / LambdaMART.ideal_dcg(
                train[train.QID == qid].Y.as_matrix())
            n_train.append(ndcg_res)

        ndcg_train.append(np.mean(n_train))
        if (len(ndcg_train)) % 5 == 0:
            plt.plot(range(1, len(ndcg_train) + 1), ndcg_train, label='train')
            plt.legend()
            plt.show()
        print('train', ndcg_train[-1])
    predictor.save("temp/mart.dump")


def get_answer():
    predictor = LambdaMART.LambdaMART()
    model_file = "temp/temp_model_660.lmart"
    logging.info(f"using {model_file}")
    predictor.load(model_file)

    test = pd.read_csv("data/testset.cvs")#, nrows=100)
    X_columns = [col for col in test.columns if col[0] == u'X']

    X = test.as_matrix(columns=X_columns)
    qids = test.QID.as_matrix()

    scores = predictor.predict(X, qids)
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
            print(f"Qid:{q}, {ind[sd[i]] + 1}, score:{scores[ind][sd[i]]}")
            f.write("{},{}\n".format(ind[sd[i]] + 1, q))


if __name__ == '__main__':
    run_num = int(open("run_number", "r").readline())
    run_num += 1
    open("run_number", "w").write("{}".format(run_num))
    print("Run number: ", run_num)
    # test()
    # prepare_data()
    # main()
    get_answer()
