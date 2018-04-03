from collections import Counter, defaultdict

import numpy as np

from DocStreamReader import DocItem
from main import read_docs, read_queries


def calc_idf():
    fields_num = len(DocItem[1])
    cnt = [Counter() for _ in range(fields_num)]

    docs_len = 0
    for doc in read_docs():
        docs_len += 1
        for i in range(fields_num):
            cnt[i].update(doc[1][i])

    for i in range(fields_num):
        cnt[i] = {k: np.log(1 + docs_len / float(v)) for k, v in cnt[i].items()}

    return cnt


def get_weighted_tf(feature_num, idf):
    # queries = read_queries()
    q_tf = defaultdict(list)

    for doc in read_docs():
        words = Counter(doc[1][feature_num])
        for c_i, q in enumerate(read_docs()):  # queries.items():
            tf = 0
            for wd in set(q[1][feature_num]):
                tf += (1 + np.log(words[wd])) * idf[wd]
            q_tf[q[0]].append(tf)
            if c_i > 20:
                break
    return q_tf


def get_tf_idf_my():
    docs_len = 20  # TODO change it
    idfs = calc_idf()
    fields_num = len(DocItem[1])

    res = []

    for i in range(fields_num):
        res.append(get_weighted_tf(i, idfs[i]))

    map_docs_to_nums = [doc.index for doc in read_docs()]

    queries = read_queries()
    for q in queries.items():
        result = np.zeros(docs_len)
        for r in res:
            result += np.array(r[q[0]])
        top_indexes = np.argsort(result)[-10:]
        for ind in reversed(top_indexes):
            print(map_docs_to_nums[ind], result[ind])
