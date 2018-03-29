import pickle

import time
from collections import OrderedDict

import numpy as np

import DocStreamReader
import sys
from os import listdir
from pymystem3 import Mystem

import gensim


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
    docs = OrderedDict()
    n = len(files)
    i = 0
    urls = read_urls()
    start = time.time()
    for doc in s_parser:
        docs[urls[doc.doc_url]] = doc
        i += 1
        if i % 10 == 0:
            sys.stdout.write(f"\n{i}/{n}, time:{time.time() - start}\n")
    pickle.dump(docs, open("documents.pickle", 'wb'))
    return docs


def load_dict():
    return pickle.load(open("documents.pickle", 'rb'))


def dump_docs_list(docs):
    pickle.dump([doc for doc in sorted(docs.items())], open("docs_in_list", "wb"))


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


# def tf_idf (docs):

if __name__ == '__main__':
    docs = prepare_dict(cut=20)
    urls = read_urls()
    queries = read_queries()

    gen_dict = gensim.corpora.Dictionary(doc.doc + doc.title for doc in docs.values())
    gen_dict.filter_extremes(no_below=2)
    gen_dict.save("gen_dict.dict")

    raw_corpus = [gen_dict.doc2bow(doc.doc + doc.title) for doc in docs.values()]
    gensim.corpora.MmCorpus.serialize('corpa.mm', raw_corpus)  # store to disk

    dictionary = gensim.corpora.Dictionary.load('gen_dict.dict')
    corpus = gensim.corpora.MmCorpus('corpa.mm')

    tfidf_model = gensim.models.TfidfModel(raw_corpus, dictionary=gen_dict)
    index_sparse = gensim.similarities.SparseMatrixSimilarity(corpus, num_features=corpus.num_terms)

    # res_file = open("sub_bm25.csv", "w")
    # res_file.write('QueryId,DocumentId\n')

    map_docs_to_nums = []
    for d in docs.items():
        map_docs_to_nums.append(d[0])
    print (map_docs_to_nums)

    for q in range (len (map_docs_to_nums)):#queries.items():
        # query_bow = gen_dict.doc2bow(q[1])
        query_tfidf = tfidf_model[gen_dict.doc2bow(docs[map_docs_to_nums[q]].doc + docs[map_docs_to_nums[q]].title)]
        index_sparse.num_best = 2
        print(index_sparse[query_tfidf])