from gensim.corpora import Dictionary
import tqdm
import sys
from gensim import corpora
from gensim import corpora, models, similarities
import numpy as np
import logging

TRY_NAME = "WORDS_TITLE"
title = True

dct = Dictionary(prune_at=None)

with open ("../data/docs.tsv") as fin:
    doc_num = 1 if title else 2
    
    for doc in tqdm.tqdm (fin):
        doc = doc.decode ("utf-8").strip ().split("\t")
        if len (doc) > doc_num:
            dct.add_documents ([doc[doc_num].split ()], prune_at=None)

dct.save ("../result/{}/dict.dct".format (TRY_NAME))

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def doc_reader (title):
    doc_num = 0 if title else 1
    
    with open ("../data/docs.tsv") as fin:
        for i, doc in enumerate (fin):
            doc = doc.decode ("utf-8").strip ().split("\t")[1:]
            if len (doc) > doc_num:
                yield dct.doc2bow(doc[doc_num].split ())
            else:
                yield []

corpora.MmCorpus.serialize('../result/{}/corpus.mm'.format (TRY_NAME), doc_reader (title))

import gensim as gs
import math
import numpy as np
import tqdm

dct_title = gs.corpora.Dictionary.load ("../result/{}/dict.dct".format (TRY_NAME))

corpus = gs.corpora.MmCorpus('../result/{}/corpus.mm'.format (TRY_NAME))

PARAM_K1 = 1.5
PARAM_B = 0.75
EPSILON = 0.25

corp_size = 0
corp_size_words = 0
for doc in tqdm.tqdm (corpus):
    corp_size_words += sum (dict (doc).itervalues ())
    corp_size += 1

avgdl = float (corp_size_words) / corp_size

idfs = {}

summ_idf = 0

for word_id, freq in dct_title.dfs.iteritems ():
    idfs[word_id] = math.log(corp_size - freq + 0.5) - math.log(freq + 0.5)
    summ_idf += idfs[word_id]

average_idf = float (summ_idf) / len (dct_title.dfs)

for id in idfs.iterkeys ():
    idfs[id] = idfs[id] if idfs[id] >= 0 else EPSILON * average_idf

q_ids = []
queries = []
with open ("../data/queries.tsv") as fin:
    for q in fin:
        q = q.strip ().decode ("utf-8").upper ().split ("\t")
        q_ids.append (int (q[0]))
        queries.append (dict (dct_title.doc2bow (q[1].split ())))

q_size = len (queries)

result = np.zeros ((q_size, corp_size))

for doc_i, doc in tqdm.tqdm (enumerate (corpus)):
    doc = dict (doc)
    doc_keys = set (doc)
    doc_len = sum (doc.itervalues ())
    for q_i, q in enumerate (queries):
        score = 0
        for word in set (q.keys ()) & doc_keys:
            idf = idfs[word]
            score += (idf * doc[word] * (PARAM_K1 + 1)
                      / (doc[word] + PARAM_K1 * (1 - PARAM_B + PARAM_B * doc_len / avgdl)))
        result[q_i, doc_i] = score

with open ("../result/result_{}.csv".format (TRY_NAME), "w") as fout:
    fout.write ("QueryId,DocumentId\n")
    
    for q_num, qid in tqdm.tqdm (enumerate (q_ids)):
        for doc in np.argsort (result[q_num, :])[-5:][::-1]:
            fout.write ("{},{}\n".format (qid, doc))

np.save (open ("../result/{}/ranking.npy".format (TRY_NAME), "wb"), result)
