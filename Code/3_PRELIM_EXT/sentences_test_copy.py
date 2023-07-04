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
mlx_meta = load_dataset("allenai/multi_lexsum", name="v20230518")
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
            # dealing with some common errors
            doc = doc.replace('; (', '; . (')
            doc2 = doc.replace('; WHER', '; . WHER')
            doc = doc2.replace('; and WHER', '; . and WHER')
            sentences = sentences + lexnlp.nlp.en.segments.sentences.get_sentence_list(doc)

    # removing common issues in segmentation
    new_sentences = []

    for sent in sentences:
        sent = sent.replace('\n', ' ')

        if len(sent.split(' ')) > 500:
            input(sent)

        # strip out just numbers and junk (prepending paras)
        if re.search('[a-zA-Z]', sent) and len(sent.split(' ')) <= 500:
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

        # get the source and tokenize
        sources_flat = '\n[DOCSPLIT]\n'.join(case['sources_clean'])
        total_tokens = tokenizer.encode(sources_flat)
        num_total_input_tokens = len(total_tokens)
        total_input_token_lengths.append(num_total_input_tokens)

        sentences = segment(sources_flat)
        num_sentences = len(sentences)
        input_sentences.append(num_sentences)

        length_issue = False
        for sentence in sentences:
            if len(sentence.split(' ')) > 500:
                length_issue = True
                break

        if length_issue is True or num_sentences > 100000:
            id = case['id']

            if length_issue is True:
                print('length')
            else:
                print('many sentneces')

            for hyp in mlx_meta[split]:
                if hyp['id'] == id:
                    print(hyp['case_metadata']['case_name'])
                    break

            for sentence in sentences:
                if len(sentence.split(' ')) > 500:
                    print('\n\n\n')
                    x = input(sentence)
                    if x.strip() == 'break':
                        break
