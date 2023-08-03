# Evaluation dump - assuming gold and pegasus summaries
# This is for a single one of the pegasus runs
# SET *** STUFF BELOW

print('Entered file!')

from rouge_score import rouge_scorer
from datasets import load_dataset, load_from_disk, load_metric
import statistics
import pandas
from collections import Counter
import math
import re

print('Imports done!')

# ***
RUN_NAME = 'data_testing'
FILE_NAME = 'outputs_pegasus_1b.csv'
CHAIN = False
RUN_TYPE = 1   # 1-5


print('** Loading results csv')
results_df = pandas.read_csv(FILE_NAME)


print('** Loading eval utils...')
scorer_rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Main function

def summary_eval(generated_output, gold_output, source, chain, RUN_TYPE):

    generated_summary = generated_output
    gold_summary = gold_output

    metrics_dict = {}

    # ROUGE based (summary)
    print('** ROUGE...')
    r1_precision, r1_recall, r1_f1 = scorer_rouge.score(gold_summary, generated_summary)['rouge1']
    metrics_dict['r1_precision'] = r1_precision
    metrics_dict['r1_recall'] = r1_recall
    metrics_dict['r1_f1'] = r1_f1

    r2_precision, r2_recall, r2_f1 = scorer_rouge.score(gold_summary, generated_summary)['rouge2']
    metrics_dict['r2_precision'] = r2_precision
    metrics_dict['r2_recall'] = r2_recall
    metrics_dict['r2_f1'] = r2_f1

    rL_precision, rL_recall, rL_f1 = scorer_rouge.score(gold_summary, generated_summary)['rougeL']
    metrics_dict['rL_precision'] = rL_precision
    metrics_dict['rL_recall'] = rL_recall
    metrics_dict['rL_f1'] = rL_f1

    return metrics_dict


def remove_nones(data):
    return [i for i in data if i is not None]


def dist_list_stats(data, title, x_label, filename):
    data = remove_nones(data)
    print('\n\n')
    print(title)
    print('MEAN')
    print(statistics.mean(data))


# MAIN


#### Actual eval metrics - do for each case and save to some data structure

results = []

for idx, case in results_df.iterrows():
    print('*** Analysing case ' + str(idx))

    gold = case['gold']
    source = case['source']
    generated = case['pegasus_output']    # oreo_pegasus_output

    # If chain or not
    metrics_dict = summary_eval(generated, gold, source, CHAIN, RUN_TYPE)
    results.append(metrics_dict)

#### Aggregrate analysis
results_df = pandas.DataFrame(results)
fn = RUN_NAME + '_metrics.csv'
results_df.to_csv(fn)

# iterate over each column
for (col_name, col_data) in results_df.iteritems():
    print('** Analysing column: ' + str(col_name))
    data = col_data.values
    fn = 'eval_outputs/' + RUN_NAME + '_dist_' + col_name
    dist_list_stats(data, str(col_name), str(col_name), fn)
