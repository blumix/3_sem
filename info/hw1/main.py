import multiprocessing
import pickle
import sys
import time
from collections import OrderedDict, defaultdict
from os import listdir

import gensim
import numpy as np
from gensim.models.doc2vec import TaggedDocument
from gensim.summarization.bm25 import BM25
from nltk.corpus import stopwords

import DocStreamReader

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
        print(scores)
        for index in reversed(max_inds[-10:]):
            print(scores[index])
            res_file.write(f"{q[0]},{map_docs_to_nums[index]}\n")
        # print(f"{i}, score{scores[reversed(max_inds[-10:])[0]]} docs.")


# def create_gensim_dict ():


def change_docs_dict(docs):
    urls = read_urls()
    f = open("new_documents.dump", "w")
    for doc in sorted(docs.items()):
        f.write(f"{urls[doc[1].doc_url]}\t{doc[1].doc_url}\t{' '.join(doc[1].title)}\t{' '.join(doc[1].doc)}\n")




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


def get_rate(ti_doc, query_bow):
    res = 0
    q_ids = [q[0] for q in query_bow]
    for id, value in ti_doc:
        if id in q_ids:
            res += value
    return res


def bm25():
    # stopwords_norm = DocStreamReader.clear_text(' '.join(stopwords.words('russian')))
    # gen_dict = gensim.corpora.Dictionary(
    #     [[w for w in extractor(doc) if w not in stopwords_norm] for doc in read_docs()], prune_at=None)
    # gen_dict.save("gen_dict.dict")
    # raw_corpus = [gen_dict.doc2bow([w for w in extractor(doc) if w not in stopwords_norm]) for doc in read_docs()]
    # gensim.corpora.MmCorpus.serialize('corpa.mm', raw_corpus)  # store to disk
    dictionary = gensim.corpora.Dictionary.load('gen_dict.dict')
    corpus = gensim.corpora.MmCorpus('corpa.mm')
    # tfidf_model = gensim.models.TfidfModel(dictionary=dictionary)#, wglobal=wglob, wlocal=wloc)
    # index_sparse = gensim.similarities.SparseMatrixSimilarity(corpus, num_features=corpus.num_terms)
    # tfidf_model.save("tf_idf.model")
    # index_sparse.save("index_sparse.matrix")
    #

    bm25 = BM25(corpus)
    average_idf = sum(map(lambda k: float(bm25.idf[k]), bm25.idf.keys())) / len(bm25.idf.keys())
    # index = gensim.similarities.SparseMatrixSimilarity(lsi[corpus])

    # index.save('lsa.index')
    # index = gensim.similarities.MatrixSimilarity.load('lsa.index')
    # tfidf_model = gensim.models.TfidfModel.load("tf_idf.model")
    # index_sparse = gensim.similarities.SparseMatrixSimilarity.load("index_sparse.matrix")
    # tf_idf_corpa = tfidf_model[corpus]

    map_docs_to_nums = pickle.load(open("map_docs_to_nums.pickle", "rb"))

    w_queries = read_queries_to_scan()

    res_file = open("sub_bm25.csv", "w")
    res_file.write('QueryId,DocumentId\n')

    # q_rates = defaultdict(list)

    for i, q in enumerate(sorted(read_queries().items())):
        query_bow = dictionary.doc2bow(q[1])
        d_i = 0
        st = np.argsort(-np.array(bm25.get_scores(query_bow, average_idf)))
        for index in st:
            if map_docs_to_nums[index] in w_queries[q[0]]:
                res_file.write(f"{q[0]},{map_docs_to_nums[index]}\n")
                d_i += 1
            if d_i == 10:
                break
        print(f"{i} docs.")


def if_idf(extractor):
    stopwords_norm = DocStreamReader.clear_text(' '.join(stopwords.words('russian')))
    gen_dict = gensim.corpora.Dictionary(
        [[w for w in extractor(doc) if w not in stopwords_norm] for doc in read_docs()], prune_at=None)
    gen_dict.save("gen_dict.dict")
    raw_corpus = [gen_dict.doc2bow([w for w in extractor(doc) if w not in stopwords_norm]) for doc in read_docs()]
    # gensim.corpora.MmCorpus.serialize('corpa.mm', raw_corpus)  # store to disk
    dictionary = gensim.corpora.Dictionary.load('gen_dict.dict')
    corpus = gensim.corpora.MmCorpus('corpa.mm')
    # tfidf_model = gensim.models.TfidfModel(dictionary=dictionary)#, wglobal=wglob, wlocal=wloc)
    # index_sparse = gensim.similarities.SparseMatrixSimilarity(corpus, num_features=corpus.num_terms)
    # tfidf_model.save("tf_idf.model")
    # index_sparse.save("index_sparse.matrix")
    #
    tfidf_model = gensim.models.TfidfModel.load("tf_idf.model")
    # index_sparse = gensim.similarities.SparseMatrixSimilarity.load("index_sparse.matrix")
    tf_idf_corpa = tfidf_model[corpus]

    map_docs_to_nums = pickle.load(open("map_docs_to_nums.pickle", "rb"))

    w_queries = read_queries_to_scan()

    res_file = open("sub_tf-idf.csv", "w")
    res_file.write('QueryId,DocumentId\n')

    q_rates = defaultdict(list)

    for i, q in enumerate(sorted(read_queries().items())):
        query_bow = dictionary.doc2bow(q[1])

        for id_num in w_queries[q[0]]:
            if id_num not in map_docs_to_nums:
                print(f"skipped {id_num}")
                continue
            ti_doc = tf_idf_corpa[map_docs_to_nums.index(id_num)]
            q_rates[q[0]].append((id_num, get_rate(ti_doc, query_bow)))

            # d_i = 0
            # st = np.argsort(-index_sparse[query_bow])
            # for index in st:
            #     if map_docs_to_nums[index] in w_queries[q[0]]:
            #         res_file.write(f"{q[0]},{map_docs_to_nums[index]}\n")
            #         d_i += 1
            #     if d_i == 10:
            #         break
        print(f"{i} docs.")
    pickle.dump(q_rates, open("q_rates.pickle", "wb"))


def read_queries_to_scan():
    w_queries = defaultdict(list)
    sub = open("sample.submission.text.relevance.spring.2018.csv", "r")
    sub.readline()
    for l in sub.readlines():
        l = l[:-1]
        spl = l.split(',')
        w_queries[int(spl[0])].append(int(spl[1]))
    return w_queries


def wglob(D, total_docs):
    return np.log(1 + D / total_docs)


def wloc(f):
    return 1 + np.log(f)




def if_idf_weighted(get_from_doc, cur_name_of_part):
    gen_dict = gensim.corpora.Dictionary([get_from_doc(doc) for doc in read_docs()], prune_at=None)
    gensim.corpora.MmCorpus.serialize(f'corpa_{cur_name_of_part}.mm',
                                      [gen_dict.doc2bow(get_from_doc(doc)) for doc in read_docs()])  # store to disk
    corpus = gensim.corpora.MmCorpus(f'corpa_{cur_name_of_part}.mm')
    tfidf_model = gensim.models.TfidfModel(dictionary=gen_dict,
                                           wglobal=wglob,
                                           wlocal=wloc)
    corpus_tfidf = tfidf_model[corpus]
    result = defaultdict(list)
    for doc_id, c_id in enumerate(corpus_tfidf):
        sys.stderr.write(f"{cur_name_of_part}:{doc_id}\n")
        for i, q in enumerate(read_queries().items()):
            query_bow = gen_dict.doc2bow(q[1])
            termid_array, tf_array = [], []
            for termid, tf in query_bow:
                termid_array.append(termid)
                tf_array.append(tf)
            tmp_calculated = np.sum([ddoc[1] for ddoc in c_id if ddoc[0] in termid_array])
            result[q[0]].append(tmp_calculated)

    pickle.dump(result, open(f"{cur_name_of_part}.dump", "wb"))
    # return result


def final_res():
    names_of_parts = ['title', 'links', 'keywords', 'description']

    coefs = [2, 1.5, 2, 2]

    result = pickle.load(open(f"text.dump", "rb"))
    # for i, part in enumerate(names_of_parts):
    #     loaded = pickle.load(open(f"{part}.dump", "rb"))
    #     for key, list_ in loaded.items():
    #         for j, val in enumerate(list_):
    #             result[key][j] += val * coefs[i]

    map_docs_to_nums = [doc.index for doc in read_docs()]

    res_file = open("sub_tf-idf_weighted.csv", "w")
    res_file.write('QueryId,DocumentId\n')

    for key, val in result.items():

        st = np.argsort(val)

        for index in range(10):
            res_file.write(f"{key},{map_docs_to_nums[st[-index]]}\n")
        print(f"{key} docs.")


def agreggate_result():
    names_of_parts = ['text', 'title', 'links', 'keywords', 'description']
    extractors = [get_from_doc_text, get_from_doc_title, get_from_doc_links, get_from_doc_keywords,
                  get_from_doc_description]

    for i in range(len(names_of_parts)):
        if_idf_weighted(extractors[i], names_of_parts[i])

    # agr_res = []
    # for i in range(len(names_of_parts)):
    #     agr_res.append(if_idf_weighted(extractors[i], names_of_parts[i]))
    #
    # pickle.dump(agr_res, open("arg_res.dump", "wb"))


def get_doc2vec():
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


# from collections import Counter, defaultdict
#
# import numpy as np
#
# from DocStreamReader import DocItem
#
# feature_num_for_docs = 2
#
#
# def calc_idf():
#     fields_num = feature_num_for_docs
#     cnt = [Counter() for _ in range(fields_num)]
#
#     docs_len = 0
#     for doc in read_docs():
#         docs_len += 1
#         for feature_name in ['title', 'body']:
#             cnt[i].update(doc[feature_name])
#
#     for i in range(fields_num):
#         cnt[i] = {k: np.log(1 + docs_len / float(v)) for k, v in cnt[i].items()}
#
#     return cnt
#
#
# def get_weighted_tf(feature_num, idf):
#     # queries = read_queries()
#     q_tf = defaultdict(list)
#
#     for doc in read_docs():
#         words = Counter(doc[1][feature_num])
#         for c_i, q in enumerate(read_docs()):  # queries.items():
#             tf = 0
#             for wd in set(q[1][feature_num]):
#                 tf += (1 + np.log(words[wd])) * idf[wd]
#             q_tf[q[0]].append(tf)
#             if c_i > 20:
#                 break
#     return q_tf
#
#
# def get_tf_idf_my():
#     docs_len = 20  # TODO change it
#     idfs = calc_idf()
#     fields_num = len(DocItem[1])
#
#     res = []
#
#     for i in range(fields_num):
#         res.append(get_weighted_tf(i, idfs[i]))
#
#     map_docs_to_nums = [doc.index for doc in read_docs()]
#
#     queries = read_queries()
#     for q in queries.items():
#         result = np.zeros(docs_len)
#         for r in res:
#             result += np.array(r[q[0]])
#         top_indexes = np.argsort(result)[-10:]
#         for ind in reversed(top_indexes):
#             print(map_docs_to_nums[ind], result[ind])



def hack_submissions():
    queries = defaultdict(list)

    sub = open("sample.submission.text.relevance.spring.2018.csv", "r")
    sub.readline()
    for l in sub.readlines():
        l = l[:-1]
        spl = l.split(',')
        queries[int(spl[0])].append(int(spl[1]))

    m_que = defaultdict(list)
    sub = open("sub_tf-idf.csv", "r")
    sub.readline()
    for l in sub.readlines():
        l = l[:-1]
        spl = l.split(',')
        m_que[int(spl[0])].append(int(spl[1]))

    res_file = open("sub_fixed_tf-idf.csv", "w")
    res_file.write('QueryId,DocumentId\n')

    for key, val in m_que.items():

        v_res = [v for v in val if v in queries[key]]
        to_add = 10 - len(v_res)

        candidates = [q for q in queries[key] if q not in v_res]

        res = v_res + np.random.choice(candidates, to_add, replace=False).tolist()
        for v in res:
            res_file.write(f"{key},{v}\n")


if __name__ == '__main__':
    # map_docs_to_nums = [doc.index for doc in read_docs()]
    # pickle.dump(map_docs_to_nums, open("map_docs_to_nums.pickle", "wb"))
    # go_parse()
    # agreggate_result()
    # final_res()
    # if_idf()
    # hack_submissions ()
    bm25()
    # for doc in read_docs():
    #     print(doc.description)

    # q_rates = pickle.load(open("q_rates.pickle", "rb"))
    #
    # res_file = open("tf_idf_finaly.csv", "w")
    # res_file.write('QueryId,DocumentId\n')
    #
    # for key, val in q_rates.items():
    #     val.sort(key=lambda tup: tup[1], reverse=True)
    #     for v in val[:10]:
    #         res_file.write(f"{key},{v[0]}\n")
    read_docs()