from gensim.corpora import Dictionary
import tqdm
import sys
from gensim import corpora, models, similarities
import numpy as np
import logging
from pymystem3 import Mystem
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from multiprocessing import Pool


def apply_to_str (string, lemmatizer, stemmer, stop):
    no_stops = filter (lambda x: x not in stop, lemmatizer.lemmatize (string))
    return map(lambda x: stemmer.stem(x), no_stops)

def process_file (fname):
    lemmatizer = Mystem()
    stemmer = SnowballStemmer('russian', ignore_stopwords=True)
    stop = stopwords.words('russian')
    stop.extend (["\n", " "])
    with open ('../data/' + fname) as f:
        with open ('../temp/' + fname + "_result", "w") as fout:
            for i, line in enumerate (f):
                splited = line.decode ("utf-8").lower ().strip ().split ("\t")
                res_str = splited[0] + "\t"
                for s in splited[1:]:
                    res_str += ' '.join (apply_to_str (s, lemmatizer, stemmer, stop)) + "\t"
                res_str += '\n'
                fout.write (res_str.encode ('utf-8'))
                if i % 1000 == 0:
                    print (fname + " at " + str (i))

files = ['xaa','xab','xac','xad','xae','xaf','xag','xah','xai','xaj','xak','xal']

pool = Pool (12)

pool.map (process_file, files)
