# plotting results from csv

import pandas
import pickle
from GRAPHS import histogram
from collections import Counter

df = pandas.read_csv('exploratory_stats.csv')

print('Means of columns')
print(df.mean(axis=0))

print('Mins of columns')
print(df.min(axis=0))

print('Maxs of columns')
print(df.max(axis=0))

print('Not none')
print(df.count())


histogram(df['number_documents'], 'Number of Documents per Case', 'Number of Documents', 'Frequency', 'num_documents_dist.png')
histogram(df['total_words_documents'], 'Number of Words in Total Case Documents', 'Number of Words', 'Frequency', 'document_total_words_dist.png')
histogram(df['long_words'], 'Number of Words in Long Summary', 'Number of Words', 'Frequency', 'long_words_dist.png')
histogram(df['long_extractive_words_ratio'], 'Ratio of Words in Long Summary which are Extractive', 'Ratio', 'Frequency', 'long_extractive_words_dist.png')
histogram(df['long_compression_ratio'], 'Long Summary Compression Ratio', 'Compression Ratio', 'Frequency', 'long_compression_dist.png')
histogram(df['short_words'], 'Number of Words in Short Summary', 'Number of Words', 'Frequency', 'short_words_dist.png')
histogram(df['short_extractive_words_ratio'], 'Ratio of Words in Short Summary which are Extractive', 'Ratio', 'Frequency', 'short_extractive_words_dist.png')
histogram(df['short_compression_ratio'], 'Short Summary Compression Ratio', 'Compression Ratio', 'Frequency', 'short_compression_dist.png')
histogram(df['long_to_short_extractive_words_ratio'], 'Ratio of Words in Short Summary which are Extractive From Long Summary', 'Ratio', 'Frequency', 'long_to_short_extractive_words_dist.png')
histogram(df['long_to_short_compression_ratio'], 'Short Summary Compression Ratio from Long Summary', 'Compression Ratio', 'Frequency', 'long_to_short_compression_dist.png')
histogram(df['tiny_words'], 'Number of Words in Tiny Summary', 'Number of Words', 'Frequency', 'tiny_words_dist.png')
histogram(df['tiny_extractive_words_ratio'], 'Ratio of Words in Tiny Summary which are Extractive', 'Ratio', 'Frequency', 'tiny_extractive_words_dist.png')
histogram(df['tiny_compression_ratio'], 'Tiny Summary Compression Ratio', 'Compression Ratio', 'Frequency', 'tiny_compression_dist.png')

with open('document_lengths.pkl', 'rb') as f:
    doc_lengths_list = pickle.load(f)

histogram(doc_lengths_list, 'Number of Words per Document', 'Number of Words', 'Frequency', 'words_per_document_dist.png')
