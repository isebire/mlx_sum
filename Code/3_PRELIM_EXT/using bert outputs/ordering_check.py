# Creating inputs based on BERT outputs for test set

import pandas
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, PegasusTokenizer
from rouge_score import rouge_scorer
import GRAPHS as graphs
import statistics
from collections import Counter
import difflib
import pprint

# Dataset for ordering -> cols are case_idx, ordered_sentences, ordered_sentences_bert
print('** Loading data...')
mlx_ordered_bert_sents = load_from_disk('ordered_tokenized_sents_per_case_test.hf')

# BERT outputs dataset -> cols are 'context', '+ve confidence', 'predicted', 'label'
bert_outputs_df = pandas.read_csv('bert_labels_test_output.csv')
sliced_col = []
for idx, row in bert_outputs_df.iterrows():
    sliced_col.append(str(row['context'])[:128])
bert_outputs_df['context_sliced'] = sliced_col

# Big dataset with ordered sources
mlx_ordered_test = load_from_disk('mlx_ordered_chains_test.hf')

# PEGASUS dataset to augment
mlx_PEGASUS_test = load_from_disk('mlx_PEGASUS_test.hf')

print(mlx_ordered_test[0]['sources_clean'])
