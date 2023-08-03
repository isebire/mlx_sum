# Creating inputs based on BERT outputs for test set
# remove ## testing parts


import pandas
from datasets import load_dataset, load_from_disk, Dataset
from transformers import AutoTokenizer, PegasusTokenizer
from rouge_score import rouge_scorer
import GRAPHS as graphs
import statistics
from collections import Counter
import difflib
import pprint
import pickle

START_FROM = 616

def get_overlap(s1, s2):
    s = difflib.SequenceMatcher(None, s1, s2)
    pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2))
    return len(s1[pos_a:pos_a+size])


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

# Dataset for ordering -> cols are case_idx, ordered_sentences, ordered_sentences_bert
print('** Loading data...')
mlx_ordered_bert_sents = load_from_disk('230714_ordered_tokenized_sents_per_case_test.hf')

# BERT outputs dataset -> cols are 'context', '+ve confidence', 'predicted', 'label'
bert_outputs_df = pandas.read_csv('bert_labels_test_output.csv')
sliced_col = []
for idx, row in bert_outputs_df.iterrows():
    sliced_col.append(str(row['context'])[:128])
bert_outputs_df['context_sliced'] = sliced_col

print('Writing file for debug')
bert_sents = bert_outputs_df['context']
with open('contexts.txt', 'w') as f:   # for testing
    for context in bert_sents:
        try:
            f.write(context)
            f.write('\n\n')
        except:
            pass


# Big dataset with ordered sources
mlx_ordered_test = load_from_disk('230714_mlx_ordered_chains_test.hf')

# PEGASUS dataset to augment
mlx_PEGASUS_test = load_from_disk('230714_mlx_PEGASUS_test.hf')

# Load utils
print('** Loading pegasus tokenizer...')
tokenizer = PegasusTokenizer.from_pretrained("nsi319/legal-pegasus")
print('** Loading rouge scorer...')
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)

print('Starting from case: ' + str(START_FROM))

# Setup data structures
if START_FROM == 0:
    augmented_test_set = []

    number_sentences_extracted = []
    number_windows_extracted = []
    number_paras_extracted = []
    tokens_sentences_extracted = []
    tokens_windows_extracted = []
    tokens_paras_extracted = []

    sentences_rouge = []
    windows_rouge = []
    paras_rouge = []
    best_rouge = []

    unmatched_sents = []
else:
    with open('augmented_test_set_progress.pkl', 'rb') as f:
        augmented_test_set = pickle.load(f)
        assert(len(augmented_test_set) == START_FROM)

    with open('number_sentences_extracted_progress.pkl', 'rb') as f:
        number_sentences_extracted = pickle.load(f)

    with open('number_windows_extracted_progress.pkl', 'rb') as f:
        number_windows_extracted = pickle.load(f)

    with open('number_paras_extracted_progress.pkl', 'rb') as f:
        number_paras_extracted = pickle.load(f)

    with open('tokens_sentences_extracted_progress.pkl', 'rb') as f:
        tokens_sentences_extracted = pickle.load(f)

    with open('tokens_windows_extracted_progress.pkl', 'rb') as f:
        tokens_windows_extracted = pickle.load(f)

    with open('tokens_paras_extracted_progress.pkl', 'rb') as f:
        tokens_paras_extracted = pickle.load(f)

    with open('sentences_rouge_progress.pkl', 'rb') as f:
        sentences_rouge = pickle.load(f)

    with open('windows_rouge_progress.pkl', 'rb') as f:
        windows_rouge = pickle.load(f)

    with open('paras_rouge_progress.pkl', 'rb') as f:
        paras_rouge = pickle.load(f)

    with open('best_rouge_progress.pkl', 'rb') as f:
        best_rouge = pickle.load(f)

    with open('unmatched_sents_progress.pkl', 'rb') as f:
        unmatched_sents = pickle.load(f)

pp = pprint.PrettyPrinter(indent=4)

for i, case in enumerate(mlx_ordered_bert_sents):
    if case['case_idx'] < START_FROM:
        continue
    print('**** Analysing case ' + str(case['case_idx']))

    case_data = {}

    # We are augmenting mlx_PEGASUS_test so copy these cols
    case_data['id'] = mlx_PEGASUS_test[i]['id']
    case_data['first_1024'] = mlx_PEGASUS_test[i]['first_1024']
    case_data['first_k'] = mlx_PEGASUS_test[i]['first_k']
    case_data['oreo_sents'] = mlx_PEGASUS_test[i]['oreo_sents']
    case_data['oreo_windows'] = mlx_PEGASUS_test[i]['oreo_windows']
    case_data['oreo_paras'] = mlx_PEGASUS_test[i]['oreo_paras']
    case_data['textrank_30'] = mlx_PEGASUS_test[i]['textrank_30']
    case_data['summary/short_no_chain'] = mlx_PEGASUS_test[i]['summary/short_no_chain']
    case_data['summary/short_chain_surface'] = mlx_PEGASUS_test[i]['summary/short_chain_surface']
    case_data['summary/short_chain_label'] = mlx_PEGASUS_test[i]['summary/short_chain_label']
    case_data['summary/short_chain_both'] = mlx_PEGASUS_test[i]['summary/short_chain_both']

    # Get gold summary -> check alignment!!!
    gold_summary = mlx_PEGASUS_test[i]['summary/short_no_chain']

    # Get ordered sources (needed for paras)
    ordered_sources = mlx_ordered_test[i]['sources_clean']
    ordered_sources_flat = '\n[DOCSPLIT]\n'.join(ordered_sources)

    # Get the scores for all the sentences in this case
    case_bert_sents = case['ordered_sentences_bert']
    ordered_sents = case['ordered_sentences']
    source_sentences_flat = ' '.join(ordered_sents)

    sents_w_non_zero_probs = []

    print('** Matching sentences (may take a while)....')

    for sentence_number, sentence in enumerate(case_bert_sents):
        # Find the corresponding line in bert_outputs_df and get the probability +ve

        if sentence.strip() == '':
            continue

        try:   # faster if there is an exact match

            #score = bert_outputs_df.loc[bert_outputs_df['context'] == sentence].iloc[0]['+ve confidence']
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
                unmatched_sents.append(sentence)
                score = 0

        # Retrive original sentence
        original_sentence = case['ordered_sentences'][sentence_number]

        # Add original sentence and score
        if score > 0:
            sents_w_non_zero_probs.append((original_sentence, score))

    # Sort sents_w_probs (as before)
    print('** Sorting sentence probabilities...')
    non_zero_sorted = sorted(sents_w_non_zero_probs, key = lambda x: x[1], reverse=True)
    final_positive = [x[0] for x in non_zero_sorted] # Sentences in order of inclusion

    # As before, create the inputs for sents / windows / paras
    # Unlike before, don't need to do postprocessing for ordering
    print('** Constructing extracts....')

    # 1. Sentences
    extracted_sentences_list = []
    for sent, score in non_zero_sorted:
        candidate = '\n'.join(extracted_sentences_list) + '\n' + sent
        if len(tokenizer.encode(candidate)) <= 1024:
            extracted_sentences_list.append(sent)
        else:
            break

    ## testing
    try:
        extracted_sentences_list = sorted([x.replace('\n', ' ') for x in extracted_sentences_list], key=lambda x: source_sentences_flat.replace('\n', ' ').index(x))
        # extracted_sentences_list = sorted([' '.join(x.replace('\n', ' ').split()) for x in extracted_sentences_list], key=lambda x: source_sentences_flat.replace('\n', ' ').index(x))
    except:
        print('!!! ERROR')
        print('sents:')
        for x in extracted_sentences_list:
            contained = x.replace('\n', ' ') in source_sentences_flat.replace('\n', ' ') #.replace('  ',' '))
            if not contained:
                print(x)
        with open('source.txt', 'w') as f:
            f.write(source_sentences_flat)
        input('info')

    number_sentences_extracted.append(len(extracted_sentences_list))
    tokens_sentences_extracted.append(len(tokenizer.encode('\n'.join(extracted_sentences_list))))
    sents_extracted = '\n'.join(extracted_sentences_list)

    # 2. Windows
    source_sentences = case['ordered_sentences']
    extracted_windows_list = []
    # used to track which sentences are there so no duplicates
    sentences_in_windows = []
    middle_sentence = []
    for sent, score in non_zero_sorted:
        sent_idx = source_sentences.index(sent)
        # Construct the window
        window = ''
        # Previous sentence, if it exists
        try:
            if source_sentences[sent_idx - 1] not in sentences_in_windows:
                window = window + source_sentences[sent_idx - 1] + ' '
                sentences_in_windows.append(source_sentences[sent_idx - 1])
        except:
            pass
        # Main sentence
        center_sentence = sent.replace('\n', ' ').strip()
        if source_sentences[sent_idx] not in sentences_in_windows:
            window = window + source_sentences[sent_idx] + ' '
            sentences_in_windows.append(source_sentences[sent_idx])
        # Following sentence, if it exists
        try:
            if source_sentences[sent_idx + 1] not in sentences_in_windows:
                window = window + source_sentences[sent_idx + 1] + ' '
                sentences_in_windows.append(source_sentences[sent_idx + 1])
        except:
            pass
        candidate = '\n'.join([x[0] for x in extracted_windows_list]) + '\n' + window
        if len(tokenizer.encode(candidate)) <= 1024:
            extracted_windows_list.append((window, center_sentence))
        else:
            break


    extracted_windows_list = sorted(extracted_windows_list, key=lambda x: source_sentences_flat.replace('\n',' ').index(x[1]))
    extracted_windows_list = [' '.join(x[0].replace('\n', ' ').split()) for x in extracted_windows_list]
    number_windows_extracted.append(len(extracted_windows_list))
    tokens_windows_extracted.append(len(tokenizer.encode('\n'.join(extracted_windows_list))))
    windows_extracted = '\n'.join(extracted_windows_list)

    # 3. Paragraphs
    source_paras_list = ordered_sources_flat.split('\n')
    extracted_paras_list = []
    for sent, score in non_zero_sorted:
        matching_paras = [para for para in source_paras_list if sent in para]
        if len(matching_paras) > 1:
            print('More than one matching para! Probs a duplicate so using first one')
        try:
            para = matching_paras[0]
            og_para = para
        except:
            print('No matching para - using overlap...')
            best_para = ''
            best_para_score = 0
            for para in source_paras_list:
                score = get_overlap(para, sent)
                if score > best_para_score:
                    best_para_score = score
                    best_para = para
            sent_words = sent.split(' ')
            contained = ''
            not_contained = ' '

            not_contained_words = False
            suffix = True
            for word in sent_words:
                if word in best_para:
                    contained = contained + word + ' '
                    if not_contained_words is True:
                        suffix = False
                        break
                else:
                    not_contained_words = True
                    not_contained = not_contained + word + ' '

            if suffix is True:
                para = best_para + not_contained
                og_para = best_para
            else:
                para = best_para
                og_para = best_para

        candidate = '\n'.join([x[0] for x in extracted_paras_list]) + '\n' + para
        if len(tokenizer.encode(candidate)) <= 1024 and para not in [x[0] for x in extracted_paras_list]:
            extracted_paras_list.append((para, og_para))
        else:
            break

    extracted_paras_list = sorted(extracted_paras_list, key=lambda x: ordered_sources_flat.replace('\n',' ').index(x[1]))
    extracted_paras_list = [' '.join(x[0].replace('\n', ' ').split()) for x in extracted_paras_list]
    number_paras_extracted.append(len(extracted_paras_list))
    tokens_paras_extracted.append(len(tokenizer.encode('\n'.join(extracted_paras_list))))
    paras_extracted = '\n'.join(extracted_paras_list)

    # Examine all the options wrt rouge
    print('** Analysing extracts....')
    current_case_scores = []

    r1_precision, r1_recall, r1_f = scorer.score(sents_extracted, gold_summary)['rouge1']
    r2_precision, r2_recall, r2_f = scorer.score(sents_extracted, gold_summary)['rouge2']
    R_mean_sent = (r1_recall + r2_recall) / 2
    sentences_rouge.append(R_mean_sent)
    current_case_scores.append((R_mean_sent, 'SENTENCES'))

    r1_precision, r1_recall, r1_f = scorer.score(windows_extracted, gold_summary)['rouge1']
    r2_precision, r2_recall, r2_f = scorer.score(windows_extracted, gold_summary)['rouge2']
    R_mean_window = (r1_recall + r2_recall) / 2
    windows_rouge.append(R_mean_window)
    current_case_scores.append((R_mean_window, 'WINDOWS'))

    r1_precision, r1_recall, r1_f = scorer.score(paras_extracted, gold_summary)['rouge1']
    r2_precision, r2_recall, r2_f = scorer.score(paras_extracted, gold_summary)['rouge2']
    R_mean_para = (r1_recall + r2_recall) / 2
    paras_rouge.append(R_mean_para)
    current_case_scores.append((R_mean_para, 'PARAGRAPHS'))

    # What is the best?
    print(current_case_scores)
    best = sorted(current_case_scores, key=lambda x: x[0], reverse=True)[0]
    print(best)
    best_rouge.append(best[1])

    case_data['bert_sents'] = sents_extracted
    case_data['bert_windows'] = windows_extracted
    case_data['bert_paras'] = paras_extracted

    # pp.pprint(case_data)

    augmented_test_set.append(case_data)

    with open('augmented_test_set_progress.pkl', 'wb') as f:
        pickle.dump(augmented_test_set, f)

    with open('number_sentences_extracted_progress.pkl', 'wb') as f:
        pickle.dump(number_sentences_extracted, f)

    with open('number_windows_extracted_progress.pkl', 'wb') as f:
        pickle.dump(number_windows_extracted, f)

    with open('number_paras_extracted_progress.pkl', 'wb') as f:
        pickle.dump(number_paras_extracted, f)

    with open('tokens_sentences_extracted_progress.pkl', 'wb') as f:
        pickle.dump(tokens_sentences_extracted, f)

    with open('tokens_windows_extracted_progress.pkl', 'wb') as f:
        pickle.dump(tokens_windows_extracted, f)

    with open('tokens_paras_extracted_progress.pkl', 'wb') as f:
        pickle.dump(tokens_paras_extracted, f)

    with open('sentences_rouge_progress.pkl', 'wb') as f:
        pickle.dump(sentences_rouge, f)

    with open('windows_rouge_progress.pkl', 'wb') as f:
        pickle.dump(windows_rouge, f)

    with open('paras_rouge_progress.pkl', 'wb') as f:
        pickle.dump(paras_rouge, f)

    with open('best_rouge_progress.pkl', 'wb') as f:
        pickle.dump(best_rouge, f)

    with open('unmatched_sents_progress.pkl', 'wb') as f:
        pickle.dump(unmatched_sents, f)


## Analyse length / tokens stats as before
dist_list_stats(number_sentences_extracted, 'Number of Sentences Extracted', 'Number of Sentences', 'number_sentences_extracted_dist')
dist_list_stats(number_windows_extracted, 'Number of Windows Extracted', 'Number of Windows', 'number_windows_extracted_dist')
dist_list_stats(number_paras_extracted, 'Number of Paragraphs Extracted', 'Number of Paragraphs', 'number_paras_extracted_dist')

dist_list_stats(tokens_sentences_extracted, 'Number of Tokens Extracted', 'Number of Tokens', 'number_tokens_sents_dist')
dist_list_stats(tokens_windows_extracted, 'Number of Tokens Extracted', 'Number of Tokens', 'number_tokens_windows_dist')
dist_list_stats(tokens_paras_extracted, 'Number of Tokens Extracted', 'Number of Tokens', 'number_tokens_paras_dist')

dist_list_stats(sentences_rouge, 'Mean of ROUGE-1 and ROUGE-2', 'Mean of ROUGE-1 and ROUGE-2', 'sents_rouge_dist')
dist_list_stats(windows_rouge, 'Mean of ROUGE-1 and ROUGE-2', 'Mean of ROUGE-1 and ROUGE-2', 'windows_rouge_dist')
dist_list_stats(paras_rouge, 'Mean of ROUGE-1 and ROUGE-2', 'Mean of ROUGE-1 and ROUGE-2', 'paras_rouge_dist')
data = [sentences_rouge, windows_rouge, paras_rouge]
graphs.box_plot(data, ['BERT - Sentences', 'BERT - Windows', 'BERT - Paragraphs'], 'Mean of ROUGE-1 and ROUGE-2 Against Gold Summary', 'Context', 'Mean of ROUGE-1 and ROUGE-2', 'rouge_dists_box_bert')

print(Counter(best_rouge))
counts_dict = dict(Counter(best_rouge))
graphs.pie_chart(counts_dict.values(), counts_dict.keys(), 'Best Performing Content Selection Granularity', 'best_option_bert', wheel=True)

## Save augmented mlx_PEGASUS_test.hf (from augmented_test_set -> df -> hf)
df = pandas.DataFrame.from_dict(augmented_test_set)
ds = Dataset.from_pandas(df)
filename = 'mlx_pegasus_w_bert_test.hf'
ds.save_to_disk(filename)
