import gensim
import numpy as np
import logging

from nltk.corpus import stopwords

import DocStreamReader as DSR

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%H:%M:%S')


def get_cosine_sim(query, docs, extractor):
    stopwords_norm = DSR.clear_text(' '.join(stopwords.words('russian')))
    dictionary = gensim.corpora.Dictionary(
        [[w for w in extractor(doc) if w not in stopwords_norm] for doc in docs], prune_at=None)

    raw_corpus = [dictionary.doc2bow([w for w in extractor(doc) if w not in stopwords_norm]) for doc in docs]
    gensim.corpora.MmCorpus.serialize(f'corpa_{query[0]}.mm', raw_corpus)
    corpus = gensim.corpora.MmCorpus(f'corpa_{query[0]}.mm')
    index_sparse = gensim.similarities.SparseMatrixSimilarity(corpus, num_features=corpus.num_terms)
    query_bow = dictionary.doc2bow(query[1])
    index_sparse.num_best = None
    return index_sparse[query_bow]


# def get_tf_idf(query, docs, extractor):
#     stopwords_norm = DSR.clear_text(' '.join(stopwords.words('russian')))
#     gen_dict = gensim.corpora.Dictionary(
#         [[w for w in extractor(doc) if w not in stopwords_norm] for doc in read_docs()], prune_at=None)
#     gen_dict.save("gen_dict.dict")
#     raw_corpus = [gen_dict.doc2bow([w for w in extractor(doc) if w not in stopwords_norm]) for doc in read_docs()]
#     # gensim.corpora.MmCorpus.serialize('corpa.mm', raw_corpus)  # store to disk
#     dictionary = gensim.corpora.Dictionary.load('gen_dict.dict')
#     corpus = gensim.corpora.MmCorpus('corpa.mm')
#     # tfidf_model = gensim.models.TfidfModel(dictionary=dictionary)#, wglobal=wglob, wlocal=wloc)
#     # index_sparse = gensim.similarities.SparseMatrixSimilarity(corpus, num_features=corpus.num_terms)
#     # tfidf_model.save("tf_idf.model")
#     # index_sparse.save("index_sparse.matrix")
#     #
#     tfidf_model = gensim.models.TfidfModel.load("tf_idf.model")
#     # index_sparse = gensim.similarities.SparseMatrixSimilarity.load("index_sparse.matrix")
#     tf_idf_corpa = tfidf_model[corpus]
#
#     map_docs_to_nums = pickle.load(open("map_docs_to_nums.pickle", "rb"))
#
#     w_queries = read_queries_to_scan()
#
#     res_file = open("sub_tf-idf.csv", "w")
#     res_file.write('QueryId,DocumentId\n')
#
#     q_rates = defaultdict(list)
#
#     for i, q in enumerate(sorted(read_queries().items())):
#         query_bow = dictionary.doc2bow(q[1])
#
#         for id_num in w_queries[q[0]]:
#             if id_num not in map_docs_to_nums:
#                 print(f"skipped {id_num}")
#                 continue
#             ti_doc = tf_idf_corpa[map_docs_to_nums.index(id_num)]
#             q_rates[q[0]].append((id_num, get_rate(ti_doc, query_bow)))
#
#             # d_i = 0
#             # st = np.argsort(-index_sparse[query_bow])
#             # for index in st:
#             #     if map_docs_to_nums[index] in w_queries[q[0]]:
#             #         res_file.write(f"{q[0]},{map_docs_to_nums[index]}\n")
#             #         d_i += 1
#             #     if d_i == 10:
#             #         break
#         print(f"{i} docs.")
#     pickle.dump(q_rates, open("q_rates.pickle", "wb"))


def one_query_job(query):
    content_types = {'title': DSR.get_from_doc_title, 'keywords': DSR.get_from_doc_keywords,
                     'links': DSR.get_from_doc_links, 'text': DSR.get_from_doc_text,
                     'description': DSR.get_from_doc_description}

    content_multipliers = {'title': 2., 'keywords': 2.,
                           'links': 0.1, 'text': 0.7,
                           'description': 1.5}

    docs_for_query = DSR.read_queries_to_scan()[query[0]]

    docs = [doc for doc in DSR.read_docs() if doc.index in docs_for_query]

    doc_ids = [doc.index for doc in docs]

    res_scores = np.zeros(len(docs))
    for key in content_types.keys():
        cur_res = np.array(get_cosine_sim(query, docs, content_types[key])) * content_multipliers[key]
        res_scores += cur_res

    best = np.argsort(-res_scores)[:10]

    return [doc_ids[best_one] for best_one in best]


def all_queries():
    global run_num
    res_file = open(f"result_{run_num}.csv", "w")
    res_file.write('QueryId,DocumentId\n')
    # documents = [doc for doc in DSR.read_docs()]
    for query in DSR.read_queries().items():
        for doc_id in one_query_job(query):
            print(query[0], doc_id)
            res_file.write(f"{query[0]},{doc_id}\n")
        logging.info(f"job for {query[0]} done.")
    res_file.close()


def docs_aggrigator():
    doc_to_query = DSR.read_doc_to_query_index()
    queries = DSR.read_queries()
    files = {qid: open(f"temp/docs_for_query_{qid}") for qid in queries.keys()}

    for doc in DSR.read_docs():
        files[doc_to_query[doc.index]].write(
            f"{doc.index}\t{doc.doc_url}\t{' '.join(doc.title)}\t{' '.join(doc.keywords)}\t{' '.join(doc.links)}\t{' '.join(doc.text)}\t{' '.join(doc.description)}\n")


def main():
    # all_queries()
    docs_aggrigator()


if __name__ == '__main__':
    run_num = int(open("run_number", "r").readline())
    run_num += 1
    open("run_number", "w").write("{}".format(run_num))
    print("Run number: ", run_num)
    main()
