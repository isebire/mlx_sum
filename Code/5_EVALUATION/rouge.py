# just rouge file for quick analysis

from rouge_score import rouge_scorer
from datasets import load_dataset, load_from_disk
import GRAPHS as graphs
import statistics
import pandas

# Still need to finish implementing metrics and data structures, and TEST!!!


def summary_eval_rouge(generated_summary, gold_summary, source):

    metrics_dict = {}

    # ROUGE based (summary)
    r1_precision, r1_recall, r1_f1 = scorer_rouge.score(gold_summary, generated_summary)['rouge1']
    metrics_dict['r1_f1'] = r1_f1

    r2_precision, r2_recall, r2_f1 = scorer_rouge.score(gold_summary, generated_summary)['rouge2']
    metrics_dict['r2_f1'] = r2_f1

    rL_precision, rL_recall, rL_f1 = scorer_rouge.score(gold_summary, generated_summary)['rougeL']
    metrics_dict['rL_f1'] = rL_f1

    return metrics_dict

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


# MAIN

RUN_NAME = 'reproduction'

print('*** Loading results csv')  # *** NOTE FILENAME AND COLUMN NAMES WILL DIFFER
results_df = pandas.read_csv('pegasus_outputs/mlx_reproduction_test.csv')


print('** Loading eval utils...')
scorer_rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


#### Actual eval metrics - do for each case and save to some data structure

results = []

for idx, case in results_df.iterrows():

    # Split into summary and entity chain if needed! !!!
    generated_summary = case['pegasus_output']
    gold_summary = case['gold']
    source = case['source']

    metrics_dict = summary_eval_rouge(generated_summary, gold_summary, source)
    results.append(metrics_dict)

#### Aggregrate analysis
results_df = pandas.DataFrame(results)
fn = RUN_NAME + '_metrics.csv'
results_df.to_csv(fn)

# iterate over each column
for (col_name, col_data) in results_df.iteritems():
    print('** Analysing column: ' + str(col_name))
    data = col_data.values
    fn = 'dist_' + col_name
    title = col_name
    x_label = col_name
    dist_list_stats(data, title, x_label, fn)
