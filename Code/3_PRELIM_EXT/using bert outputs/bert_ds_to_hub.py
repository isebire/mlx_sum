# Reorder MLX-CLEAN with documents in temporal order
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
import pandas
import requests
import re
from bs4 import BeautifulSoup
import time
from dateutil.parser import parse

mlx_pegasus_wbert = DatasetDict()

for split in ['validation', 'test', 'train']:

    if split == 'test':
        fn = 'mlx_pegasus_w_bert_test.hf'
    else:
        fn = '230714_mlx_pegasus_' + split + '.hf'
    ds = load_from_disk(fn)

    if split != 'test':
        aug_ds = []
        for i, case in enumerate(ds):
            case_data = {}
            case_data['id'] = case['id']
            case_data['first_1024'] = case['first_1024']
            case_data['first_k'] = case['first_k']
            case_data['oreo_sents'] = case['oreo_sents']
            case_data['oreo_windows'] = case['oreo_windows']
            case_data['oreo_paras'] = case['oreo_paras']
            case_data['textrank_30'] = case['textrank_30']
            case_data['summary/short_no_chain'] = case['summary/short_no_chain']
            case_data['summary/short_chain_surface'] = case['summary/short_chain_surface']
            case_data['summary/short_chain_label'] = case['summary/short_chain_label']
            case_data['summary/short_chain_both'] = case['summary/short_chain_both']
            case_data['bert_sents'] = ''
            case_data['bert_windows'] = ''
            case_data['bert_paras'] = ''
            aug_ds.append(case_data)
        df = pandas.DataFrame.from_dict(aug_ds)
        ds = Dataset.from_pandas(df)

    mlx_pegasus_wbert[split] = ds


# output the DatasetDict huggingface
print('*** Uploading to huggingface')
mlx_pegasus_wbert.push_to_hub("isebire/mlx_PEGASUS_WBERT", private=True)
