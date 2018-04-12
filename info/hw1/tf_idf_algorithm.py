import gensim
import numpy as np
import logging

from nltk.corpus import stopwords

import DocStreamReader as DSR

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%H:%M:%S')

logging.disable(logging.INFO)


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


def wglob(D, total_docs):
    return np.log(1 + D / total_docs)


def wloc(f):
    return 1 + np.log(f)


def get_rate(ti_doc, query_bow):
    res = 0
    q_ids = [q[0] for q in query_bow]
    for id, value in ti_doc:
        if id in q_ids:
            res += value
    return res


def get_tf_idf(query, docs, extractor):
    stopwords_norm = DSR.clear_text(' '.join(stopwords.words('russian')))
    dictionary = gensim.corpora.Dictionary(
        [[w for w in extractor(doc) if w not in stopwords_norm] for doc in docs], prune_at=None)

    raw_corpus = [dictionary.doc2bow([w for w in extractor(doc) if w not in stopwords_norm]) for doc in docs]
    gensim.corpora.MmCorpus.serialize(f'temp/corpa_{query[0]}.mm', raw_corpus)
    corpus = gensim.corpora.MmCorpus(f'temp/corpa_{query[0]}.mm')
    tfidf_model = gensim.models.TfidfModel(dictionary=dictionary)#, wglobal=wglob, wlocal=wloc)
    tf_idf_corpa = tfidf_model[corpus]

    query_bow = dictionary.doc2bow(query[1])

    scores = []
    for ti_doc in tf_idf_corpa:
        scores.append(get_rate(ti_doc, query_bow))
    return scores


def one_query_job(query):
    content_types = {'title': DSR.get_from_doc_title, 'keywords': DSR.get_from_doc_keywords,
                     'links': DSR.get_from_doc_links, 'text': DSR.get_from_doc_text,
                     'description': DSR.get_from_doc_description}

    content_multipliers = {'title': 2.0, 'keywords': 0.3,
                           'links': 0.1, 'text': 1,
                           'description': 0.6}

    docs = [doc for doc in DSR.read_docs_for_query(query[0])]

    doc_ids = [doc.index for doc in docs]

    res_scores = np.zeros(len(docs))
    for key in content_types.keys():
        cur_res_tf = np.array(get_tf_idf(query, docs, content_types[key])) * content_multipliers[key]
        cur_res_cos = np.array(get_cosine_sim(query, docs, content_types[key])) * content_multipliers[key] * 2
        cur_res = cur_res_cos + cur_res_tf
        #print(cur_res.min(), cur_res.max())
        res_scores += cur_res

    best = np.argsort(-res_scores)[:10]

    return [doc_ids[best_one] for best_one in best]


def all_queries():
    global run_num
    res_file = open(f"result/result_{run_num}.csv", "w")
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
    files = {qid: open(f"temp/docs_for_query_{qid}", "w") for qid in queries.keys()}

    for doc in DSR.read_docs():
        files[doc_to_query[doc.index]].write(
            f"{doc.index}\t{doc.url}\t{' '.join(doc.title)}\t{' '.join(doc.keywords)}\t{' '.join(doc.links)}\t{' '.join(doc.text)}\t{' '.join(doc.description)}\n")


def main():
    all_queries()
    # docs_aggrigator()


if __name__ == '__main__':
    run_num = int(open("run_number", "r").readline())
    run_num += 1
    open("run_number", "w").write("{}".format(run_num))
    print("Run number: ", run_num)
    main()
