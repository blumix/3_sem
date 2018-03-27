import pickle
import sys
import time

import multiprocessing
import re
import testfixtures as testfixtures
from bs4 import BeautifulSoup
from gensim import corpora, models
from os import listdir, path
from pymystem3 import Mystem
from six import iteritems
from nltk.corpus import stopwords
import logging
from gensim.models.doc2vec import TaggedDocument
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class doc_reader:
    def __init__(self, path, stemm, urls):
        with open(path) as f:
            html = ""
            try:
                self.doc_num = urls[f.readline().rstrip()]
                html = f.read()
            except:
                self.doc_num = -1
                pass

        soup = BeautifulSoup(html, "html.parser")

        for script in soup(["script", "style"]):
            script.decompose()  # rip it out
        body = soup.find('body')
        title = soup.find('title')
        if body is None:
            body = soup.get_text()
        else:
            body = body.get_text()

        self.body = self.clear_text(body.lower(), stemm)
        if title is not None:
            self.title = self.clear_text(title.get_text().lower(), stemm)
        pass

    def clear_text(self, text, stemm):
        patt = re.compile(r'[\w]+', flags=re.UNICODE)
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        text = '\n'.join(patt.findall(text))
        return [word for word in stemm.lemmatize(text) if patt.search(word)]

    def get_tokens(self):
        if not hasattr(self, 'title'):
            self.title = []
        return self.title + self.body


class docs_container:
    def __init__(self, path, stemm, urls):
        self.path = path
        self.stemm = stemm
        self.urls = urls
        self.files = [path + "/" + f for f in listdir(path)]
        self.docs = {}

    def read_docs(self):
        start_time = time.time()
        for i, f_name in enumerate(self.files):
            doc = doc_reader(f_name, self.stemm, self.urls)
            self.docs[doc.doc_num] = doc
            if i % 10 == 0:
                sys.stderr.write(f"path:{self.path} | {i}/{len(self.files)} | time:{time.time() - start_time}\n")
                sys.stderr.flush()
        self.stemm = None
        self.urls = None
        self.files = None


def clear_text(text, stemm):
    patt = re.compile(r'[\w]+', flags=re.UNICODE)
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    text = '\n'.join(patt.findall(text))
    return [word for word in stemm.lemmatize(text) if patt.search(word)]


def read_queries(f_name):
    urls = {}
    stemm = Mystem(entire_input=True)

    with open(f_name) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    for url in content:
        split = url.split('\t')
        urls[int(split[0])] = clear_text(split[1], stemm)
    return urls


def load_models():
    model_dm = models.doc2vec.Doc2Vec.load("model_dm.model")
    model_dbow = models.doc2vec.Doc2Vec.load("model_dbow.model")
    return model_dm, model_dbow


def get_model(docs):
    # MODEL PARAMETERS
    dm = 1  # 1 for distributed memory(default); 0 for dbow
    cores = multiprocessing.cpu_count()
    size = 160
    context_window = 10
    seed = 42
    min_count = 2
    alpha = 0.5
    max_iter = 150

    # BUILD MODEL
    if None:
        model_dm = models.doc2vec.Doc2Vec(dm=dm,
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

        model_dbow = models.doc2vec.Doc2Vec(dm=0, size=size, negative=5, hs=0, min_count=min_count, workers=cores,
                                            epochs=max_iter)
        model_dm.build_vocab(
            documents=[TaggedDocument(doc.get_tokens(), [doc.doc_num]) for doc in docs.values() if doc.doc_num != -1])
        model_dbow.reset_from(model_dm)
        model_dm.save("model_dm.model")
        model_dbow.save("model_dbow.model")
        model_dm.train(
            documents=[TaggedDocument(doc.get_tokens(), [doc.doc_num]) for doc in docs.values() if doc.doc_num != -1],
            total_examples=len(docs), epochs=max_iter)
        model_dm.save("model_dm.model")
    model_dm, model_dbow = load_models()
    model_dbow.train(
        documents=[TaggedDocument(doc.get_tokens(), [doc.doc_num]) for doc in docs.values() if doc.doc_num != -1],
        total_examples=len(docs), epochs=max_iter)
    model_dbow.save("model_dbow.model")
    return model_dm, model_dbow


def main():
    path_to_folder = "content/"
    files = [f for f in listdir(path_to_folder) if path.isfile(path_to_folder + f)]
    first = pickle.load(open(path_to_folder + files[0], 'rb'))
    for f in files[1:]:
        temp = pickle.load(open(path_to_folder + f, 'rb'))
        first.docs.update(temp.docs)

    model_dm, model_dbow = get_model(first.docs)

    querys = read_queries("queries.numerate.txt")
    # concat_model = ConcatenatedDoc2Vec([model_dm, model_dbow])
    # model_dm, model_dbow = load_models()
    models = [model_dm, model_dbow]  # , concat_model]
    files = [open("sub_dm.csv", "w"), open("sub_dbow.csv", "w"), open("sub_concat.csv", "w")]

    for i in range(2):
        files[i].write('QueryId,DocumentId\n')
        for q in querys.items():
            new_vector = models[i].infer_vector(q[1])
            sims = models[i].docvecs.most_similar([new_vector])
            for s in sims:
                files[i].write(f"{q[0]},{s[0]}\n")
        files[i].close()


if __name__ == "__main__":
    main()
