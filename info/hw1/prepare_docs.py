import pickle
import time
import re
from multiprocessing import Pool
from os import listdir, os

import sys
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from pymystem3 import Mystem
from sklearn.feature_extraction.text import TfidfVectorizer


def read_urls(f_name):
    urls = {}
    with open(f_name) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    for url in content:
        split = url.split('\t')
        urls[split[1]] = int(split[0])
    return urls


def read_queries(f_name):
    urls = {}
    patt = re.compile(r"[\w]+", flags=re.UNICODE)
    with open(f_name) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    for url in content:
        split = url.split('\t')
        urls[int(split[0])] = set([word for word in set(patt.split(split[1].lower())) if
                                   word not in stopwords.words('russian')])
        if u'' in urls[int(split[0])]:
            urls[int(split[0])].remove(u'')
    return urls


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


def scan_dir(dir_name):
    urls = read_urls("urls.numerate.txt")
    stemm = Mystem(entire_input=True)
    stemm.start()
    d_c = docs_container(dir_name, stemm, urls)
    d_c.read_docs()
    pickle.dump(d_c, open(dir_name + "scanned.dump", "wb"))


class all_doc_reader:
    def __init__(self, path):
        self.path = path
        self.dirs = [path + d for d in listdir(path)]
        self.n_pools = len(self.dirs)

    def run(self):
        # scan_dir(self.dirs[3])
        with Pool(len(self.dirs)) as p:
            p.map(scan_dir, self.dirs)
        #     # scan_dir(d)


def main_1():
    reader = all_doc_reader("content/")
    reader.run()


def main():
    path = "content/"
    files = [f for f in os.listdir(path) if os.path.isfile(f)]
    first = pickle.load(files[0])
    for f in files[1:]:
        temp = pickle.load(f)
        first.docs.update(temp.docs)
    print ("here")
    vectorizer = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x.body + x.title)
    vectorizer.fit_transform(first.docs)
    idf = vectorizer.idf_
    idf_result = dict(zip(vectorizer.get_feature_names(), idf))
    pickle.dump(idf_result, open ("idf.dump", "wb"))


if __name__ == "__main__":
    main()
