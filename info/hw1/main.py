import pickle

import time

import multiprocessing
from collections import OrderedDict

import numpy as np
from gensim.models.doc2vec import TaggedDocument

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


def get_bm25():
    dictionary = gensim.corpora.Dictionary.load('gen_dict.dict')
    corpus = gensim.corpora.MmCorpus('corpa.mm')
    bm25 = gensim.summarization.bm25.BM25(corpus)
    av_idf = np.sum(i for i in bm25.idf.values()) / len(bm25.idf)
    map_docs_to_nums = [doc.index for doc in read_docs()]

    # for i, doc in enumerate (read_docs()):
    #     scores = bm25.get_scores(dictionary.doc2bow(doc.title + doc.body), av_idf)
    #     max_inds = np.argsort(scores)
    #     print (f"{doc.index}, {map_docs_to_nums[max_inds[0]]}, , {map_docs_to_nums[max_inds[-1]]}")
    #     if i > 20:
    #         break

    res_file = open("sub_bm25.csv", "w")
    res_file.write('QueryId,DocumentId\n')

    for i, q in enumerate(read_queries().items()):
        query_bow = dictionary.doc2bow(q[1])
        scores = bm25.get_scores(query_bow, av_idf)
        max_inds = np.argsort(scores)
        for index in reversed(max_inds[-10:]):
            res_file.write(f"{q[0]},{map_docs_to_nums[index]}\n")
        print(f"{i} docs.")

# def create_gensim_dict ():


def change_docs_dict(docs):
    urls = read_urls()
    f = open("new_documents.dump", "w")
    for doc in sorted(docs.items()):
        f.write(f"{urls[doc[1].doc_url]}\t{doc[1].doc_url}\t{' '.join(doc[1].title)}\t{' '.join(doc[1].doc)}\n")


class document:
    def __init__(self, line):
        spl = line.split("\t")
        self.index = int(spl[0])
        self.url = spl[1]
        self.title = spl[2].lower().split(' ')
        self.body = spl[3].lower().split(' ')


def read_docs():
    f = open("new_documents.dump", 'r')

    for line in f.readlines():
        yield document(line)


def get_model_doc2vec():
    # MODEL PARAMETERS
    dm = 1  # 1 for distributed memory(default); 0 for dbow
    cores = multiprocessing.cpu_count()
    size = 300
    context_window = 20
    seed = 42
    min_count = 2
    alpha = 0.5
    max_iter = 100

    # BUILD MODEL
    model_dm = gensim.models.doc2vec.Doc2Vec(
        documents=[TaggedDocument(doc.title + doc.body, [doc.index]) for doc in read_docs()],
        dm=dm,
        window=context_window,
        alpha=alpha,
        seed=seed,
        min_count=min_count,  # ignore words with freq less than min_count
        max_vocab_size=None,  #
        vector_size=size,  # is the dimensionality of the feature vector
        sample=5e-6,  # ?
        negative=5,  # ?
        workers=cores,  # number of cores
        epochs=max_iter  # number of iterations (epochs) over the corpus
    )
    model_dm.save("model_dm.model")
    model_dm = gensim.models.doc2vec.Doc2Vec.load("model_dm.model")
    return model_dm


def if_idf():
    # print(f"number of docs is:{sum(1 for _ in read_docs())}")
    # docs = prepare_dict(cut=20)
    queries = read_queries()
    #
    gen_dict = gensim.corpora.Dictionary(doc.title + doc.body for doc in read_docs())
    gen_dict.filter_extremes(no_below=1, no_above=1, keep_n=None)
    gen_dict.save("gen_dict.dict")

    raw_corpus = [gen_dict.doc2bow(doc.title + doc.body) for doc in read_docs()]
    gensim.corpora.MmCorpus.serialize('corpa.mm', raw_corpus)  # store to disk

    dictionary = gensim.corpora.Dictionary.load('gen_dict.dict')
    corpus = gensim.corpora.MmCorpus('corpa.mm')

    tfidf_model = gensim.models.TfidfModel(raw_corpus, dictionary=dictionary)
    index_sparse = gensim.similarities.SparseMatrixSimilarity(corpus, num_features=corpus.num_terms)
    tfidf_model.save("tf_idf.model")
    index_sparse.save("index_sparse.matrix")

    tfidf_model = gensim.models.TfidfModel.load("tf_idf.model")
    index_sparse = gensim.similarities.SparseMatrixSimilarity.load("index_sparse.matrix")

    map_docs_to_nums = [doc.index for doc in read_docs()]

    res_file = open("sub_bm25.csv", "w")
    res_file.write('QueryId,DocumentId\n')

    for i, q in enumerate(queries.items()):
        query_bow = dictionary.doc2bow(q[1])
        query_tfidf = tfidf_model[query_bow]
        index_sparse.num_best = 10
        for index in index_sparse[query_tfidf]:
            res_file.write(f"{q[0]},{map_docs_to_nums[index[0]]}\n")
        print(f"{i} docs.")



def get_doc2vec ():
    # model = get_model_doc2vec()
    model = gensim.models.doc2vec.Doc2Vec.load("model_dm.model")

    # res_file = open("sub_doc2vec.csv", "w")
    # res_file.write('QueryId,DocumentId\n')

    map_docs_to_nums = [doc.index for doc in read_docs()]

    for i, q in enumerate(read_docs()):  # queries.items():
        new_vector = model.infer_vector(q.body + q.title)
        sims = model.docvecs.most_similar([new_vector])
        for s in sims:
            print(f"{s}, {map_docs_to_nums[s[0]]}")
        print("\n")
        if i > 10:
            break
            # res_file.write(f"{q[0]},{s[0]}\n")
    # res_file.close()

if __name__ == '__main__':
    get_bm25 ()

