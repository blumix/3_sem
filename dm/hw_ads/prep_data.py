import sys
# import tqdm


def go_run():
    f = open("train.csv")
    tr_file = open("train_my.svm", "w")
    keys = f.readline().strip().split(";")
    norm_features = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18]
    gr_features = [12, 13, 14]
    num = 0
    for l in f.readlines():
        num += 1
        if num % 10000 == 0:
            sys.stdout.write("\r{}".format(num))
        pl = l.strip().split(";")
        res_str = "{} ".format(pl[1])
        for i in norm_features:
            if pl[i] is not '':
                res_str += "{}:{} ".format(i, pl[i])
        for g in gr_features:
            for p in pl[g].split(','):
                if p is not '':
                    res_str += "{}:1 ".format(str(g) + "_" + str(p))
        res_str += "\n"
        tr_file.write(res_str)


# def go_run_test():
#     f = open("test.csv")
#     te_file = open("test_my.vw", "w")
#     keys = f.readline().strip().split(";")
#     norm_features = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18]
#     gr_features = [12, 13, 14]
#     num = 0
#     for l in f.readlines():
#         num += 1
#         #         print (l)
#         if num % 10000 == 0:
#             sys.stdout.write("\r{}".format(num))
#         pl = l.strip().split(";")
#         res_str = "{} |n ".format(pl[1])
#         for i in norm_features:
#             if pl[i] is not '':
#                 res_str += "{}:{} ".format(i, pl[i])
#         for g in gr_features:
#             for p in pl[g].split(','):
#                 if p is not '':
#                     res_str += "{} ".format(str(g) + "_" + str(p))
#         res_str += "\n"
#         te_file.write(res_str)


go_run()
# go_run_test()
