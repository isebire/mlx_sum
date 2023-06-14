# counting number of sentences in short summary

from datasets import load_dataset
import pandas
import lexnlp.nlp.en.segments.sentences
from pprint import pprint
import statistics
from collections import Counter
import GRAPHS as graphs
import numpy as np

mlx_clean = load_dataset('isebire/multi_lexsum_CLEANED_AGAIN')

short_summary_num_sentences = []

for split in ['train', 'validation', 'test']:

    for case in mlx_clean[split]:
        if case['summary/short_w_chain'] is not None:
            summary = case['summary/short_w_chain'].split('[SUMMARY]')[1].strip()
            sentences = lexnlp.nlp.en.segments.sentences.get_sentence_list(summary)

            short_summary_num_sentences.append(len(sentences))

# output mean min max
print('Number of sentences in short summary: max mean min')
print(max(short_summary_num_sentences))
print(statistics.mean(short_summary_num_sentences))
print(min(short_summary_num_sentences))

print('90th percentile')
p = np.percentile(short_summary_num_sentences, 90) # return 90th percentile
print(p)

print('Counts')
print(Counter(short_summary_num_sentences))

# print dist
graphs.histogram(short_summary_num_sentences, 'Number of Sentences in Short Summary', 'Number of Sentences', 'Frequency', 'short_summary_num_sentences', log_y=False)
