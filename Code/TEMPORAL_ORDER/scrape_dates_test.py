# dates

import requests
import re
from bs4 import BeautifulSoup
import pickle
import pandas
import time
from dateutil.parser import parse
#from datasets import load_dataset

doc_links = ['https://clearinghouse.net/doc/31854', 'https://clearinghouse.net/doc/55245', 'https://clearinghouse.net/doc/61795', 'https://clearinghouse.net/doc/65505', 'https://clearinghouse.net/doc/30551']
temporal_order = []

#mlx_clean = load_dataset('isebire/mlx_CLEAN_FINAL')
#mlx_meta = load_dataset("allenai/multi_lexsum", name="v20230518")

# ping the webpage
for original_position, url in enumerate(doc_links):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    r = requests.get(url, headers=headers).content
    soup = BeautifulSoup(r, features="lxml")
    div = soup.find("div", {"class": "flex-grow-1"})
    date = str(div).split("<h1", 1)[1].split('(')[1].split(')')[0]
    time.sleep(0.5)
    temporal_order.append((original_position, parse(date)))

temporal_order = sorted(temporal_order, key=lambda x: x[1])
temporal_indices = [x[0] for x in temporal_order]
print(temporal_indices)
# docket last
temporal_indices.append(temporal_indices.pop(0))
print(temporal_indices)
