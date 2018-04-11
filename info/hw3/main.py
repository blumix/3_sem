import pandas as pd
import numpy as np

import LambdaMART


def main():
    train = pd.read_csv("data/train.data.cvs")
    X_columns = [col for col in train.columns if col[0] == u'X']
    train_colums = ['QID', 'Y'].extend(X_columns)
    train_X = train.as_matrix(columns=train_colums)

    predictor = LambdaMART.LambdaMART(training_data=train_X)
    predictor.fit()

    test = pd.read_csv("data/testset.cvs")
    test_columns = ['QID'].extend(X_columns)

    scores = predictor.predict(test.as_matrix(columns=test_columns))
    qid = set(test.QID)
    global run_num
    f = open("results/try_{}.csv".format(run_num), "w")
    f.write("DocumentId,QueryId\n")
    for q in sorted(qid):
        ind = test[test.QID == q].index

        print(q)

        sd = np.argsort(scores[ind])
        lem = len(sd)
        if lem > 5:
            lem = 5
        for i in range(lem):
            f.write("{}, {}\n".format(ind[sd[i]], q))


if __name__ == '__main__':
    run_num = int(open("run_number", "r").readline())
    run_num += 1
    open("run_number", "w").write("{}".format(run_num))

    main()
