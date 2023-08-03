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

verbose = False

def original_stats(gold_summary, generated_summary):
    metrics_dict = {}

    r1_precision, r1_recall, r1_f1 = scorer_rouge.score(gold_summary, generated_summary)['rouge1']
    metrics_dict['r1_prec'] = r1_precision

    r2_precision, r2_recall, r2_f1 = scorer_rouge.score(gold_summary, generated_summary)['rouge2']
    metrics_dict['r2_prec'] = r2_precision

    rL_precision, rL_recall, rL_f1 = scorer_rouge.score(gold_summary, generated_summary)['rougeL']
    metrics_dict['rL_prec'] = rL_precision
    return metrics_dict


def dist_list_stats(data, title, x_label, filename):
    #print('MIN')
    #print(min(data))
    print('MEAN: ' + str(statistics.mean(data)))
    #print('MAX')
    #print(max(data))
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


results = {}
for extractive_output_col in extract_cols:
    results[extractive_output_col] = {'rouge_1_precs': [], 'rouge_2_precs': [], 'rouge_L_precs': [], 'R_means_precs': []}

# For each case
best_option = []
sentences_best = []
windows_best = []
paras_best = []



for index, case in enumerate(mlx_test):

    print('* Analysing case ' + str(index))

    gold = case['summary/short_no_chain']

    R_mean_options = []

    for extractive_output_col in extract_cols:
        print(extractive_output_col)
        extract = case[extractive_output_col]
        stats_results = original_stats(gold, extract)
        results[extractive_output_col]['rouge_1_precs'].append(stats_results['r1_prec'])
        results[extractive_output_col]['rouge_2_precs'].append(stats_results['r2_prec'])
        results[extractive_output_col]['rouge_L_precs'].append(stats_results['rL_prec'])
        if verbose:
            print(stats_results)

        R_mean = (stats_results['r1_prec'] + stats_results['r2_prec']) / 2
        results[extractive_output_col]['R_means_precs'].append(R_mean)
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
    if results['bert_sents']['R_means_precs'][-1] > results['oreo_sents']['R_means_precs'][-1]:
        if verbose:
            print('bert better')
        sentences_best.append('BERT')
    else:
        if verbose:
            print('oreo better')
        sentences_best.append('OREO')

    if results['bert_windows']['R_means_precs'][-1] > results['oreo_windows']['R_means_precs'][-1]:
        if verbose:
            print('bert better')
        windows_best.append('BERT')
    else:
        if verbose:
            print('oreo better')
        windows_best.append('OREO')

    if results['bert_paras']['R_means_precs'][-1] > results['oreo_paras']['R_means_precs'][-1]:
        if verbose:
            print('bert better')
        paras_best.append('BERT')
    else:
        if verbose:
            print('oreo better')
        paras_best.append('OREO')


# Output the aggregrate stats

metric_lists = ['rouge_1_precs', 'rouge_2_precs', 'rouge_L_precs', 'R_means_precs']
metric_titles = {'rouge_1_precs': 'ROUGE-1 Precision', 'rouge_2_precs': 'ROUGE-2 Precision', 'rouge_L_precs': 'Rouge-L Precision', 'R_means_precs': 'Mean of ROUGE-1 Precision and ROUGE-2 Precision'}
for metric in metric_lists:
    for extractive_strat in extract_cols:
        print(extractive_strat)
        print(metric)
        # Means and histograms for each metric and each strategy
        fn = 'PREC_' + metric + '_' + extractive_strat + '_dist'
        dist_list_stats(results[extractive_strat][metric], metric_titles[metric], metric_titles[metric], fn)

    # Box plot for R_mean for main 6 strats    ## CONTINUE FROM HERE AND TEST!!!
    data = [results['first_1024'][metric], results['first_k'][metric], results['bert_sents'][metric], results['bert_windows'][metric], results['bert_paras'][metric], results['textrank_30'][metric]]
    fn = 'PREC_6_box_' + metric
    graphs.box_plot(data, ['First 1024', 'First K', 'BERT - Sentences', 'BERT - Windows', 'BERT - Paragraphs', 'TextRank'], metric_titles[metric], 'Extractive Strategy', metric_titles[metric], fn, long_labels=True)

    # Box plot for BERT vs OREO
    #data = [results['bert_sents'][metric], results['oreo_sents'][metric], results['bert_windows'][metric], results['oreo_windows'][metric], results['bert_paras'][metric], results['oreo_paras'][metric]]
    #fn = 'PREC_oreo_bert_pairs_box_' + metric
    #graphs.box_plot(data, ['BERT - Sentences', 'OREO - Sentences', 'BERT - Windows', 'OREO - Windows', 'BERT - Paragraphs', 'OREO - Paragraphs'], metric_titles[metric], 'Extractive Strategy', metric_titles[metric], fn, positions=True, long_labels=True)



# output stats for which of pair for each 3 pairs is better
print('OREO vs BERT - sentences')
print(Counter(sentences_best))
print('OREO vs BERT - windows')
print(Counter(windows_best))
print('OREO vs BERT - paragraphs')
print(Counter(paras_best))


print('Best option:')
best_option_dict = dict(Counter(best_option))
graphs.pie_chart(best_option_dict.values(), best_option_dict.keys(), 'Best Option in Terms of Mean of ROUGE-1 Precision and ROUGE-2 Precision', 'PREC_best_R_prec')
