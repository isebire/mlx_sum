# Reorder MLX-CLEAN with documents in temporal order
from datasets import load_dataset, Dataset, DatasetDict
import pandas
import requests
import re
from bs4 import BeautifulSoup
import time
from dateutil.parser import parse

mlx_clean = load_dataset('isebire/mlx_CLEAN_FINAL')
mlx_meta = load_dataset("allenai/multi_lexsum", name="v20230518")

mlx_temporal = DatasetDict()

for split in ['validation']:  #['validation', 'test', 'train']:

    split_data = []


    for i, case in enumerate(mlx_clean[split]):

        if case['id'] == 'PC-CO-0002':
            print(len(case['sources_clean']))
            for doc in case['sources_clean']:
                print('start')
                print(doc[:200])
                print('end')
                print(doc[-200:])
                input('    ?   ')
            input('')
        else:
            continue
