import pandas as pd
import numpy as np
import LambdaMART_1 as LambdaMART
import logging
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def prepare_data():
    train = pd.read_csv("data/train.data.cvs")#, nrows=2000)
    logging.info("data was read.")
    not_ranked = [qid for qid in set(train.QID) if len(set(train.Y[train.QID == qid])) == 1]
    for qid in not_ranked:
        train = train.drop(train[train.QID == qid].index, axis=0)
    train = train.reset_index(drop=True)

    for qid in set(train.QID):
        for num, val, in enumerate(sorted(set(train.Y[train.QID == qid]))):
            train.loc[(train.QID == qid) & (train.Y == val), 'Y'] = num

    train = train.sort_values(by=['QID', 'Y'], ascending=False)
    train = train.reset_index(drop=True)
    train.to_csv("data/prepared_train.csv")


def main():
    train = pd.read_csv("data/prepared_train.csv")#, nrows=100)
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
    train_colums = ['Y', 'QID'] + X_columns

    #predictor = LambdaMART.LambdaMART(training_data=train.as_matrix(columns=train_colums), number_of_trees=100,
    #                                 learning_rate=0.1, max_depth=4)
    predictor = LambdaMART.LambdaMART(number_of_trees=1000)
    predictor.load('temp/temp_model_90.lmart')
    predictor.learning_rate = 1
    predictor.number_of_trees = 1500
    ndcg_train = []
    for scores_train in predictor.fit():
        n_train = []
        for qid in set(train.QID):
            ind = train[train.QID == qid].index

            my_sort = np.argsort(scores_train[ind])
            ndcg_res = LambdaMART.dcg_k(train[train.QID == qid].Y.as_matrix()[my_sort],
                                        5) / LambdaMART.ideal_dcg_k(
                train[train.QID == qid].Y.as_matrix(), 5)
            n_train.append(ndcg_res)

        ndcg_train.append(np.mean(n_train))
        if (len(ndcg_train)) % 10 == 0:
            plt.plot(range(1, len(ndcg_train) + 1), ndcg_train, label='train')
            plt.legend()
            plt.show()
        print(ndcg_train[-1])

    predictor.save("temp/mart.dump")
    get_answer()


def get_answer():
    predictor = LambdaMART.LambdaMART()
    predictor.load("temp/temp_model_2000.lmart")

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


if __name__ == '__main__':
    run_num = int(open("run_number", "r").readline())
    run_num += 1
    open("run_number", "w").write("{}".format(run_num))
    print("Run number: ", run_num)
    # test()
    prepare_data()
    main()
    # get_answer()
    # counstruct_data_for_xgb()
    # run_xgboost()
