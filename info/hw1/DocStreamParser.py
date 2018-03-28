import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from pymystem3 import Mystem

patt = re.compile(r'[^\W]+', flags=re.UNICODE)
stemm = Mystem(entire_input=False)


def clear_text(text):
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    text = '\n'.join(patt.findall(text))
    return [word for word in stemm.lemmatize(text) if patt.search(word) and word not in stopwords.words('russian')]


class DocumentStreamParser:
    def __init__(self, paths):
        self.paths = paths

    def __iter__(self):
        for path in self.paths:
            doc = {}
            with open(path) as f:
                try:
                    doc['url'] = f.readline().rstrip()
                    html = f.read()
                except:
                    continue

                soup = BeautifulSoup(html, "html.parser")

                for script in soup(["script", "style"]):
                    script.decompose()  # rip it out
                body = soup.find('body')
                title = soup.find('title')
                if body is None:
                    body = soup.get_text()
                else:
                    body = body.get_text()

                doc['body'] = clear_text(body.lower())
                if title is not None:
                    doc['title'] = clear_text(title.get_text().lower())
                yield doc
