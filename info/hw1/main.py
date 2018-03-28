import pickle

import time

import numpy as np

import DocStreamReader
import sys
from os import listdir
from pymystem3 import Mystem

from gensim import summarization


def get_all_dat_files(folder):
    f_folders = [folder + f for f in listdir(folder)]
    files = []
    for fold in f_folders:
        files.extend([fold + '/' + f for f in listdir(fold)])
    return files


def read_queries(f_name='queries.numerate.txt'):
    queries = {}
    with open(f_name, encoding='utf-8') as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    for url in content:
        split = url.split('\t')
        queries[int(split[0])] = DocStreamReader.clear_text(split[1])
    return queries


def read_urls(f_name='urls.numerate.txt'):
    urls = {}
    with open(f_name) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    for url in content:
        split = url.split('\t')
        urls[split[1]] = int(split[0])
    return urls


def prepare_dict(cut=None):
    files = get_all_dat_files('content/')
    if cut:
        files = files[:cut]
    s_parser = DocStreamReader.load_files_multiprocess(files)
    docs = {}
    n = len(files)
    i = 0
    start = time.time()
    for doc in s_parser:
        docs[doc.doc_url] = doc
        i += 1
        if i % 10 == 0:
            sys.stdout.write(f"\n{i}/{n}, time:{time.time() - start}\n")
    pickle.dump(docs, open("documents.pickle", 'wb'))


def load_dict():
    return pickle.load(open("documents.pickle", 'rb'))


def dump_docs_list(docs):
    pickle.dump ([doc for doc in sorted(docs.items())], open ("docs_in_list", "wb"))


# def get_bm25():
#     global st, q, s
#     bm25 = summarization.bm25.BM25(get_corp(docs_dict))
#     av_idf = np.sum(i for i in bm25.idf.values()) / len(bm25.idf)
#     res_file = open("sub_bm25.csv", "w")
#     res_file.write('QueryId,DocumentId\n')
#     st = sorted(docs_dict.keys())
#     for q in queries.items():
#         scores = bm25.get_scores(q[1], av_idf)
#         max_inds = np.argsort(scores)[:2]
#
#         for s in max_inds:
#             res_file.write(f"{q[0]},{urls[st[s]]}\n")
#     res_file.close()
#
# def create_gensim_dict ():


if __name__ == '__main__':
    # prepare_dict(cut=5)
    # urls = read_urls()
    # queries = read_queries()
    docs_dict = load_dict()
    dump_docs_list(docs_dict)
    # for d in docs_dict.items():
    #     print(f"url in doc:{d[1].doc_url} url_num:{urls[d[1].doc_url]}")
    #     print(d[1].title)
    #     print(d[1].doc)
    #     break
