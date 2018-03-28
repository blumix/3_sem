import re
from collections import namedtuple
from nltk.corpus import stopwords
from pymystem3 import Mystem

TRACE_NUM = 1000
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%H:%M:%S')


def trace(items_num, trace_num=TRACE_NUM):
    if items_num % trace_num == 0: logging.info("Complete items %05d" % items_num)


def trace_worker(items_num, worker_id, trace_num=TRACE_NUM):
    if items_num % trace_num == 0: logging.info("Complete items %05d in worker_id %d" % (items_num, worker_id))


def html2text_bs_visible(raw_html):
    from bs4 import BeautifulSoup
    """
    Тут производится извлечения из html текста, который видим пользователю
    """
    soup = BeautifulSoup(raw_html, "html.parser")
    [s.extract() for s in soup(['style', 'script', '[document]'])]

    try:
        title = soup.find('title').get_text()
    except:
        title = ''

    [s.extract() for s in soup(['title', 'head'])]
    body = soup.get_text()
    return title, body


patt = re.compile(r'[^\W]+', flags=re.UNICODE)
stemm = Mystem(entire_input=False)


def clear_text(text):
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    text = '\n'.join(patt.findall(text))
    return [word for word in stemm.lemmatize(text) if patt.search(word) and word not in stopwords.words('russian')]


html2text = html2text_bs_visible


def toks(title, body):
    return clear_text(title), clear_text(body)


def html2word(raw_html, to_text=html2text):
    title, body = to_text(raw_html)
    return toks(title, body)


from multiprocessing import Process, Queue

DocItem = namedtuple('DocItem', ['doc_url', 'title', 'doc'])

WORKER_NUM = 8


def load_csv_worker(files, worker_id, res_queue):
    for i, file in enumerate(files):
        if i % WORKER_NUM != worker_id: continue
        with open(file,encoding='utf-8') as input_file:
            trace_worker(i, worker_id)
            print (file)
            try:
                url = input_file.readline().rstrip()
                html = input_file.read()
            except:
                continue

            title, body = html2word(html)

            res_queue.put(DocItem(url, title, body))

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
