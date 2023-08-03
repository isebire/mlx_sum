# Reorder MLX-CLEAN with documents in temporal order
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
import pandas
import requests
import re
from bs4 import BeautifulSoup
import time
from dateutil.parser import parse

mlx_temporal_2 = DatasetDict()

for split in ['validation', 'test', 'train']:

    fn = '230714_mlx_cleaned_ordered_' + split + '.hf'
    ds = load_from_disk(fn)

    mlx_temporal_2[split] = ds


# output the DatasetDict huggingface
print('*** Uploading to huggingface')
mlx_temporal_2.push_to_hub("isebire/230714_mlx_CLEANED_ORDERED", private=True)
