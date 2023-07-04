# Need to work out the number of sentences that can fit into encoder

# Calculate avg number of tokens per source sentence

# OR do percentages. ratio: number of tokens in sources / number of tokens in encoder = num sents in source / num sents in encoder
# but need a flat # of sentences for all

# This script also formats MLX for OREO as it is the sentence segmentation
# that takes a long time
# Need to have a seperate json file for each of the splits w/ the following
#	text (ESSENTIAL): a list, where each element is a string of that sentence
#	summary (ESSENTIAL): list, where each element is a sentence


from datasets import load_dataset, Dataset
import pandas
import lexnlp.nlp.en.segments.sentences
from pprint import pprint
import statistics
from collections import Counter
import GRAPHS as graphs
import numpy as np
from transformers import AutoTokenizer, PegasusTokenizer
import re

mlx_clean = load_dataset('isebire/mlx_CLEAN_FINAL')
tokenizer = PegasusTokenizer.from_pretrained("nsi319/legal-pegasus")

ENCODER_MAX_TOKENS = 1024
total_input_token_lengths = []
sentence_token_lengths = []
input_sentences = []
est_encoder_sentences_list = []

def segment(text):
    docs = text.split('\n[DOCSPLIT]\n')
    sentences = []
    for doc in docs:
        if 'CIVIL DOCKET' in doc:
            sentences = sentences + doc.split('\n')
        else:
            sentences = sentences + lexnlp.nlp.en.segments.sentences.get_sentence_list(doc)

    # removing common issues in segmentation
    new_sentences = []

    for sent in sentences:
        sent = sent.replace('\n', ' ')

        # strip out just numbers (prepending paras)
        if re.search('[a-zA-Z]', sent):
            # fixing common segmentation errors
            # merge with next if previous sentence ends v.
            if new_sentences != []:

                if new_sentences[-1].strip().endswith('v.'):
                    new_sentences[-1] = new_sentences[-1] + ' ' + sent

                # merge with previous if current starts Section
                if sent.startswith('Section'):
                    new_sentences[-1] = new_sentences[-1] + ' ' + sent

                else:
                    new_sentences.append(sent)

            else:
                new_sentences.append(sent)

    return new_sentences


for split in ['train', 'validation', 'test']:

    split_data = []

    for i, case in enumerate(mlx_clean[split]):
        print('Case ' + str(i))

        if case['summary/short_w_chain'] is None:
            continue

        # this will hold the row of dataset to be converted into json for oreo
        case_data = {}

        # get the source and tokenize
        sources_flat = '\n[DOCSPLIT]\n'.join(case['sources_clean'])
        total_tokens = tokenizer.encode(sources_flat)
        num_total_input_tokens = len(total_tokens)
        total_input_token_lengths.append(num_total_input_tokens)

        sentences = segment(sources_flat)
        num_sentences = len(sentences)
        input_sentences.append(num_sentences)

        for sentence in sentences:
            sentence_tokens = tokenizer.encode(sentence)
            sentence_token_lengths.append(len(sentence_tokens))

        # adding int so it rounds down
        est_encoder_sentences = int((ENCODER_MAX_TOKENS / num_total_input_tokens) * num_sentences)
        est_encoder_sentences_list.append(est_encoder_sentences)

        # Saving oreo format
        # for s in sentences:
        #     print(s)
        case_data['text'] = sentences

        summary = case['summary/short_w_chain'].split('[SUMMARY]')[1].strip()
        summary_sentences = segment(summary)
        case_data['summary'] = summary_sentences

        split_data.append(case_data)


    # SAVE SPLIT AS JSON FOR OREO
    # Make the list of dicts into a pandas dataframe
    df = pandas.DataFrame.from_dict(split_data)

    # Make the pandas dataframe into a hugging face dataset
    ds = Dataset.from_pandas(df)

    # Export to json
    filename = 'mlx_clean_oreo_' + split + '.json'
    ds.to_json(filename)


# output mean min max
print('Number of tokens for sources overall: max mean min')
print(max(total_input_token_lengths))
print(statistics.mean(total_input_token_lengths))
print(min(total_input_token_lengths))

print('90th percentile')
p = np.percentile(total_input_token_lengths, 90) # return 90th percentile
print(p)

# print dist
graphs.histogram(total_input_token_lengths, 'Number of Tokens for Sources Overall', 'Number of Tokens', 'Frequency', 'source_overall_tokens', log_y=False)


print('Number of tokens per source sentence: max mean min')
print(max(sentence_token_lengths))
print(statistics.mean(sentence_token_lengths))
print(min(sentence_token_lengths))

print('90th percentile')
p = np.percentile(sentence_token_lengths, 90) # return 90th percentile
print(p)

# print dist
graphs.histogram(sentence_token_lengths, 'Number of Tokens per Source Sentence', 'Number of Tokens', 'Frequency', 'source_sentence_tokens', log_y=False)


print('Estimated num of sentences to extract for encoder : max mean min')
print(max(est_encoder_sentences_list))
print(statistics.mean(est_encoder_sentences_list))
print(min(est_encoder_sentences_list))

print('90th percentile')
p = np.percentile(est_encoder_sentences_list, 90) # return 90th percentile
print(p)

# print dist
graphs.histogram(est_encoder_sentences_list, 'Estimated Number of Sentences To Extract', 'Number of Sentences', 'Frequency', 'num_sentences_extract', log_y=False)

print('Number of sentences in sources for case: max mean min')
print(max(input_sentences))
print(statistics.mean(input_sentences))
print(min(input_sentences))

print('90th percentile')
p = np.percentile(input_sentences, 90) # return 90th percentile
print(p)

# print dist
graphs.histogram(input_sentences, 'Number of Sentences in Source', 'Number of Sentences', 'Frequency', 'sentences_in_source', log_y=False)
