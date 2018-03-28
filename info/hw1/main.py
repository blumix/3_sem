import pickle

import time

import sys
from os import listdir
import DocStreamReader


def get_all_dat_files(folder):
    f_folders = [folder + f for f in listdir(folder)]
    files = []
    for fold in f_folders:
        files.extend([fold + '/' + f for f in listdir(fold)])
    return files


if __name__ == '__main__':
    files = get_all_dat_files('content/')
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
