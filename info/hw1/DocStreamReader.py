import re
from collections import namedtuple
from nltk.corpus import stopwords
from pymystem3 import Mystem
from bs4 import BeautifulSoup

TRACE_NUM = 1000
import logging

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
