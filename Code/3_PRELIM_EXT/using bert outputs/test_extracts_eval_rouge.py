import pandas
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PegasusTokenizer
from rouge_score import rouge_scorer
from datasets import load_dataset, load_from_disk, load_metric
import GRAPHS as graphs
import statistics
from collections import Counter
import re
from rouge_score import rouge_scorer
import pickle
import argparse
import time

START_FROM = 0
verbose = False

def original_stats(gold_summary, generated_summary):
    metrics_dict = {}

    r1_precision, r1_recall, r1_f1 = scorer_rouge.score(gold_summary, generated_summary)['rouge1']
    metrics_dict['r1_f1'] = r1_f1

    r2_precision, r2_recall, r2_f1 = scorer_rouge.score(gold_summary, generated_summary)['rouge2']
    metrics_dict['r2_f1'] = r2_f1

    rL_precision, rL_recall, rL_f1 = scorer_rouge.score(gold_summary, generated_summary)['rougeL']
    metrics_dict['rL_f1'] = rL_f1

    if verbose:
        print(r1_f1)
        print(r2_f1)
        print(rL_f1)

    return metrics_dict


def dist_list_stats(data, title, x_label, filename):
    print('MIN')
    print(min(data))
    print('MEAN')
    print(statistics.mean(data))
    print('MAX')
    print(max(data))
    graphs.histogram(data, title, x_label, 'Frequency', filename, bin_num=20, log_y=False)

# Load ORDERED VALIDATION dataset as hugginface dataset
print('** Loading dataset HF...')
mlx_test = load_from_disk('mlx_pegasus_w_bert_test.hf')

# Load utils
scorer_rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

extract_cols = ['first_1024',
                'first_k',
                'bert_sents', 'oreo_sents',
                'bert_windows', 'oreo_windows',
                'bert_paras', 'oreo_paras',
                'textrank_30']

# Need to save these for each of the 9 strats
# rouge_1_f1s = []
# rouge_2_f1s = []
# rouge_L_f1s = []
# R_means = []
# bs_f1s = []


if START_FROM == 0:
    results = {}
    for extractive_output_col in extract_cols:
        results[extractive_output_col] = {'rouge_1_f1s': [], 'rouge_2_f1s': [], 'rouge_L_f1s': [], 'R_means': []}

    # For each case
    best_option = []
    sentences_best = []
    windows_best = []
    paras_best = []

else:
    with open('extract_eval_progress.pkl', 'rb') as f:
        results = pickle.load(f)

    with open('extract_best_progress.pkl', 'rb') as f:
        best_option = pickle.load(f)
        assert(len(best_option) == START_FROM)

    with open('extract_sentences_best_progress.pkl', 'rb') as f:
        sentences_best = pickle.load(f)

    with open('extract_windows_best_progress.pkl', 'rb') as f:
        windows_best = pickle.load(f)

    with open('extract_paras_best_progress.pkl', 'rb') as f:
        paras_best = pickle.load(f)


for index, case in enumerate(mlx_test):
    if index < START_FROM:
        continue

    print('* Analysing case ' + str(index))

    gold = case['summary/short_no_chain']

    R_mean_options = []

    for extractive_output_col in extract_cols:
        print(extractive_output_col)
        extract = case[extractive_output_col]
        stats_results = original_stats(gold, extract)
        results[extractive_output_col]['rouge_1_f1s'].append(stats_results['r1_f1'])
        results[extractive_output_col]['rouge_2_f1s'].append(stats_results['r2_f1'])
        results[extractive_output_col]['rouge_L_f1s'].append(stats_results['rL_f1'])
        if verbose:
            print(stats_results)

        R_mean = (stats_results['r1_f1'] + stats_results['r2_f1']) / 2
        results[extractive_output_col]['R_means'].append(R_mean)
        if verbose:
            print(R_mean)

        if extractive_output_col not in ['oreo_sents', 'oreo_windows', 'oreo_paras']:
            R_mean_options.append((R_mean, extractive_output_col))


    # What is the best option for each case based on R_mean
    print('Based option for case based on R means?')
    best = sorted(R_mean_options, key=lambda x: x[0], reverse=True)[0]
    print(best)
    best_option.append(best[1])

    # For each of the 3 pairs is bert or oreo better?
    if results['bert_sents']['R_means'][-1] > results['oreo_sents']['R_means'][-1]:
        if verbose:
            print('bert better')
        sentences_best.append('BERT')
    else:
        if verbose:
            print('oreo better')
        sentences_best.append('OREO')

    if results['bert_windows']['R_means'][-1] > results['oreo_windows']['R_means'][-1]:
        if verbose:
            print('bert better')
        windows_best.append('BERT')
    else:
        if verbose:
            print('oreo better')
        windows_best.append('OREO')

    if results['bert_paras']['R_means'][-1] > results['oreo_paras']['R_means'][-1]:
        if verbose:
            print('bert better')
        paras_best.append('BERT')
    else:
        if verbose:
            print('oreo better')
        paras_best.append('OREO')

    # Pickle dumps
    with open('extract_eval_progress.pkl', 'wb') as f:
        pickle.dump(results, f)

    with open('extract_best_progress.pkl', 'wb') as f:
        pickle.dump(best_option, f)

    with open('extract_sentences_best_progress.pkl', 'wb') as f:
        pickle.dump(sentences_best, f)

    with open('extract_windows_best_progress.pkl', 'wb') as f:
        pickle.dump(windows_best, f)

    with open('extract_paras_best_progress.pkl', 'wb') as f:
        pickle.dump(paras_best, f)


# Output the aggregrate stats

metric_lists = ['rouge_1_f1s', 'rouge_2_f1s', 'rouge_L_f1s', 'R_means']
metric_titles = {'rouge_1_f1s': 'ROUGE-1 F1', 'rouge_2_f1s': 'ROUGE-2 F1', 'rouge_L_f1s': 'Rouge-L F1', 'R_means': 'Mean of ROUGE-1 F1 and ROUGE-2 F1'}
for metric in metric_lists:
    for extractive_strat in extract_cols:
        print(extractive_strat)
        print(metric)
        # Means and histograms for each metric and each strategy
        fn = metric + '_' + extractive_strat + '_dist'
        dist_list_stats(results[extractive_strat][metric], metric_titles[metric], metric_titles[metric], fn)

    # Box plot for R_mean for main 6 strats    ## CONTINUE FROM HERE AND TEST!!!
    data = [results['first_1024'][metric], results['first_k'][metric], results['bert_sents'][metric], results['bert_windows'][metric], results['bert_paras'][metric], results['textrank_30'][metric]]
    fn = '6_box_' + metric
    graphs.box_plot(data, ['First 1024', 'First K', 'BERT - Sentences', 'BERT - Windows', 'BERT - Paragraphs', 'TextRank'], metric_titles[metric], 'Extractive Strategy', metric_titles[metric], fn, long_labels=True)

    # Box plot for BERT vs OREO
    data = [results['bert_sents'][metric], results['oreo_sents'][metric], results['bert_windows'][metric], results['oreo_windows'][metric], results['bert_paras'][metric], results['oreo_paras'][metric]]
    fn = 'oreo_bert_pairs_box_' + metric
    graphs.box_plot(data, ['BERT - Sentences', 'OREO - Sentences', 'BERT - Windows', 'OREO - Windows', 'BERT - Paragraphs', 'OREO - Paragraphs'], metric_titles[metric], 'Extractive Strategy', metric_titles[metric], fn, positions=True, long_labels=True)



# output stats for which of pair for each 3 pairs is better
print('OREO vs BERT - sentences')
print(Counter(sentences_best))
print('OREO vs BERT - windows')
print(Counter(windows_best))
print('OREO vs BERT - paragraphs')
print(Counter(paras_best))
