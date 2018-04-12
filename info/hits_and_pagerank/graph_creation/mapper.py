#!/usr/bin/env python
import sys
import base64
import zlib
from bs4 import BeautifulSoup, SoupStrainer

sys.path.append('.')
with open("urls.txt", 'r') as f:
    urls_inv = {int ( line[:-1].split ('\t')[1]):  line[:-1].split ('\t')[0] for line in f.readlines ()}

for line in sys.stdin:
    line = line.strip ().split ('\t')
    line[1] = line[1][:-1]
    html = zlib.decompress (base64.b64decode (line[1]))
    soup = BeautifulSoup (html)
    for link in soup.find_all('a', href=True):
        try:
            if link['href'][0] == '/':
                print "{}\t{}".format (line[0], urls_inv['http://lenta.ru' + link['href']])
            else:
                print "{}\t{}".format (line[0], urls_inv[link['href']])
        except:
            pass