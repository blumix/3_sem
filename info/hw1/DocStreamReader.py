import re
import sys
from collections import namedtuple, defaultdict
from datetime import time
from os import listdir

from bs4 import BeautifulSoup
from pymystem3 import Mystem
import logging

TRACE_NUM = 1000

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%H:%M:%S')


def trace(items_num, trace_num=TRACE_NUM):
    if items_num % trace_num == 0: logging.info("Complete items %05d" % items_num)


def trace_worker(items_num, worker_id, trace_num=TRACE_NUM):
    if items_num % trace_num == 0: logging.info("Complete items %05d in worker_id %d" % (items_num, worker_id))


def html2text_bs_visible(raw_html):
    """
    Тут производится извлечения из html текста, который видим пользователю
    """
    soup = BeautifulSoup(raw_html, "html.parser")

    for s in soup.findAll(['script', 'style']):
        s.decompose()

    try:
        links = u' '.join(link.extract().get_text().lower() for link in soup('a'))
    except:
        links = u''
    try:
        keywords = u' '.join(tag.attrs['content'] for tag in soup('meta') if
                             'name' in tag.attrs.keys() and tag.attrs['name'].strip().lower() in ['keywords'])
    except:
        keywords = u''

    try:
        description = u' '.join(tag.attrs['content'] for tag in soup('meta') if
                                'name' in tag.attrs.keys() and tag.attrs['name'].strip().lower() in [
                                    'description'])
    except:
        description = u''

    for s in soup('meta'): s.decompose()
    try:
        title = u' '.join(link.extract().get_text().lower() for link in soup('title'))
    except:
        title = u''
    try:
        body = soup.getText()
    except:
        body = u''

    return title, keywords, links, body, description


patt = re.compile(r'[^\W]+', flags=re.UNICODE)
stemm = Mystem(entire_input=False)


def clear_text(text):
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    text = '\n'.join(patt.findall(text))
    return [word for word in stemm.lemmatize(text) if patt.search(word)]


html2text = html2text_bs_visible


def toks(to_clear):
    return clear_text(to_clear)


def html2word(raw_html, to_text=html2text):
    fields = to_text(raw_html)
    return [toks(f) for f in fields]


from multiprocessing import Process, Queue

DocItem = namedtuple('DocItem', ['doc_url', 'title', 'keywords', 'links', 'text', 'description'])

WORKER_NUM = 12


def load_csv_worker(files, worker_id, res_queue):
    for i, file in enumerate(files):
        if i % WORKER_NUM != worker_id: continue
        with open(file, encoding='utf-8') as input_file:
            trace_worker(i, worker_id)
            try:
                url = input_file.readline().rstrip()
                html = input_file.read()
            except:
                continue

            res_queue.put(DocItem(url, *html2word(html)))

        trace_worker(i, worker_id, 1)
    res_queue.put(None)


def load_files_multiprocess(input_file_name):
    processes = []
    res_queue = Queue()
    for i in range(WORKER_NUM):
        process = Process(target=load_csv_worker, args=(input_file_name, i, res_queue))
        processes.append(process)
        process.start()

    complete_workers = 0
    while complete_workers != WORKER_NUM:
        item = res_queue.get()
        if item is None:
            complete_workers += 1
        else:
            yield item

    for process in processes: process.join()


class Document:
    def __init__(self, line):
        spl = line.split("\t")
        self.index = int(spl[0])
        self.url = spl[1]
        self.title = spl[2].lower().split(' ')
        self.keywords = spl[3].lower().split(' ')
        self.links = spl[4].lower().split(' ')
        self.text = spl[5].lower().split(' ')
        self.description = spl[6][:-1].lower().split(' ')


def read_docs():
    f = open("temp/new_documents.dump", 'r', encoding='utf-8')

    # i = 0
    for line in f.readlines():
        # sys.stderr.write(f"\r{i} doc read.")
        # i += 1
        yield Document(line)


def get_all_dat_files(folder):
    f_folders = [folder + f for f in listdir(folder)]
    files = []
    for fold in f_folders:
        files.extend([fold + '/' + f for f in listdir(fold)])
    return files


def go_parse():
    urls = read_urls()
    f = open("temp/new_documents.dump", "w", encoding='utf-8')

    start = time.time()
    files = get_all_dat_files('content/')
    s_parser = load_files_multiprocess(files)
    for doc in s_parser:
        f.write(
            f"{urls[doc.doc_url]}\t{doc.doc_url}\t{' '.join(doc.title)}\t{' '.join(doc.keywords)}\t{' '.join(doc.links)}\t{' '.join(doc.text)}\t{' '.join(doc.description)}\n")
    f.close()
    print(time.time() - start)


def read_queries(f_name='data/queries.numerate.txt'):
    queries = {}
    with open(f_name, encoding='utf-8') as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    for url in content:
        split = url.split('\t')
        queries[int(split[0])] = clear_text(split[1])
    return queries


def read_urls(f_name='data/urls.numerate.txt'):
    urls = {}
    with open(f_name) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    for url in content:
        split = url.split('\t')
        urls[split[1]] = int(split[0])
    return urls


def read_queries_to_scan():
    w_queries = defaultdict(list)
    sub = open("data/sample.submission.text.relevance.spring.2018.csv", "r")
    sub.readline()
    for l in sub.readlines():
        l = l[:-1]
        spl = l.split(',')
        w_queries[int(spl[0])].append(int(spl[1]))
    return w_queries


def read_doc_to_query_index():
    docs = {}
    sub = open("data/sample.submission.text.relevance.spring.2018.csv", "r")
    sub.readline()
    for l in sub.readlines():
        l = l[:-1]
        spl = l.split(',')
        docs[int(spl[1])] = int(spl[0])
    return docs


def get_from_doc_title(doc):
    return doc.title


def get_from_doc_text(doc):
    return doc.text


def get_from_doc_links(doc):
    return doc.links


def get_from_doc_keywords(doc):
    return doc.keywords


def get_from_doc_description(doc):
    return doc.description
