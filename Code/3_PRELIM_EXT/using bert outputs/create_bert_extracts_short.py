# Creating inputs based on BERT outputs for test set
# remove ## testing parts


import pandas
from datasets import load_dataset, load_from_disk, Dataset
from transformers import AutoTokenizer, PegasusTokenizer, BertTokenizer
import statistics
from collections import Counter
import pickle

def dist_list_stats(data, title, x_label, filename):
    print('\n\n')
    print(title)
    print('MIN')
    print(min(data))
    print('MEAN')
    print(statistics.mean(data))
    print('MAX')
    print(max(data))
    graphs.histogram(data, title, x_label, 'Frequency', filename, bin_num=20, log_y=False)

# BERT outputs dataset -> cols are 'context', '+ve confidence', 'predicted', 'label'
bert_outputs_df = pandas.read_csv('bert_labels_test_output.csv')
sliced_col = []
for idx, row in bert_outputs_df.iterrows():
    sliced_col.append(str(row['context'])[:128])
bert_outputs_df['context_sliced'] = sliced_col

# PEGASUS dataset to augment
mlx_PEGASUS_test = load_from_disk('mlx_pegasus_w_bert_test.hf')

# Load utils
print('** Loading pegasus tokenizer...')
tokenizer = PegasusTokenizer.from_pretrained("nsi319/legal-pegasus")

tokens_sentences_extracted = []
augmented_test_set = []

tokenizer_bert = BertTokenizer.from_pretrained('casehold/custom-legalbert', do_lower_case=True)

for i, case in enumerate(mlx_PEGASUS_test):
    print('**** Analysing case ' + str(i))

    case_data = case

    # Get the sentences in the BERT summary
    bert_sentences_extract = case['bert_sents']
    bert_sents = bert_sentences_extract.split('\n')

    bert_sents_tokenized = []
    for sent in bert_sents:
        bert_sents_tokenized.append(' '.join(tokenizer_bert.tokenize(sent)[:-1]))

    # Get the score for each of these sentences
    sents_w_non_zero_probs = []
    for sentence, og_sentence in zip(bert_sents_tokenized, bert_sents):
        try:   # faster if there is an exact match
            score = bert_outputs_df[bert_outputs_df['context'] == sentence].iloc[0]['+ve confidence']
        except:
            # Need to slice both if no match -> sentence may be longer due to
            # bert truncation.
            sentence_sliced = sentence[:128]
            try:
                #score = bert_outputs_df.loc[bert_outputs_df['context_sliced'] == sentence_sliced].iloc[0]['+ve confidence']
                score = bert_outputs_df[bert_outputs_df['context_sliced'] == sentence_sliced].iloc[0]['+ve confidence']
            except:
                print('!!!! UNMATCHED SENT')
                print(sentence)
                score = 0

        if score > 0:
            sents_w_non_zero_probs.append((og_sentence, score))

    # Sort so top scoring
    non_zero_sorted = sorted(sents_w_non_zero_probs, key = lambda x: x[1], reverse=True)

    # Limit the number of sentences
    oreo_summary = case['oreo_sents']
    oreo_sents = len(oreo_summary.split('\n'))
    max_sents = int(oreo_sents * 1.13) + 1
    non_zero_sorted = non_zero_sorted[:max_sents]
    included_sents = [x[0] for x in non_zero_sorted]

    # In same order as orignal BERT sumamry, only keep if in truncated list :)
    new_sentences = []
    for sentence in bert_sents:
        if sentence in included_sents:
            new_sentences.append(sentence)

    extracted_sentences_list = new_sentences

    # Save
    tokens_sentences_extracted.append(len(tokenizer.encode('\n'.join(extracted_sentences_list))))
    sents_extracted = '\n'.join(extracted_sentences_list)
    case_data['bert_sents_short'] = sents_extracted

    augmented_test_set.append(case_data)



## Save augmented mlx_PEGASUS_test.hf (from augmented_test_set -> df -> hf)
df = pandas.DataFrame.from_dict(augmented_test_set)
ds = Dataset.from_pandas(df)
filename = 'mlx_pegasus_short_bert_sents_test.hf'
ds.save_to_disk(filename)
