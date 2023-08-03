# Evaluation dump - assuming gold and pegasus summaries
# This is for a single one of the pegasus runs
# SET *** STUFF BELOW

print('Entered file!')

from rouge_score import rouge_scorer
from datasets import load_dataset, load_from_disk, load_metric
import GRAPHS as graphs
import statistics
from bert_score import score
import pandas
import ner_tag_2 as ner
from collections import Counter
import math
import re

print('Imports done!')

# ***
RUN_NAME = 'testing'
FILE_NAME = 'pegasus_outputs/outputs_pegasus_2.csv'
CHAIN = False
RUN_TYPE = 2   # 1-5

# Still need to finish implementing metrics and data structures, and TEST!!!
hallucinated_entity_types = []

print('** Loading results csv')
results_df = pandas.read_csv(FILE_NAME)


print('** Loading eval utils...')
scorer_rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
bertscore = load_metric("bertscore")   # move

# Main function

def summary_eval(generated_output, gold_output, source, chain, RUN_TYPE):

    # Split into chain and summary if needed
    if chain:
        malformed_chain = False
        # split by [SUMMARY] if exists
        gold_chain, gold_summary = gold_output.split('[SUMMARY]')
        gold_chain = gold_chain.strip('[ENTITYCHAIN]')

        if '[SUMMARY]' in generated_output:
            generated_chain, generated_summary = generated_output.split('[SUMMARY]')
            generated_chain = generated_chain.strip('[ENTITYCHAIN]')
        else:
            generated_summary = generated_output
            malformed_chain = True
    else:
        generated_summary = generated_output
        gold_summary = gold_output

    metrics_dict = {}

    if chain:
        if malformed_chain:
            metrics_dict['malformed_chain'] = 1
        else:
            metrics_dict['malformed_chain'] = 0

    # Record of all columns - starred is reported in OG
    # malformed_chain (if chain one)
    # r1_precision, r1_recall, *r1_f1
    # r2_precision, r2_recall, *r2_f1
    # rL_precision, rL_recall, *rL_f1
    # bs_precision, bs_recall, *bs_f1
    # bs_mnli_precision, bs_mnli_recall, bs_mnli_f1
    # unique_trigram_ratio
    # nid
    # redundancy
    # grammatical_errors
    # pegasus_entailment, gold_entailment
    # pegasus_flesch_kincaid, pegasus_coleman_liau, pegasus_ari, pegasus_smog
    # gold_flesch_kincaid, gold_coleman_liau, gold_ari, gold_smog

    ## SUMMARY EFFECTIVENESS

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

    print(r1_f1)
    print(r2_f1)
    print(rL_f1)

    # BERTScore for comparison with original mlx paper
    print('** BERTScore...')

    bertscore_results = bertscore.compute(predictions=[generated_summary], references=[gold_summary], lang="en", model_type="microsoft/deberta-large-mnli", rescale_with_baseline=True, use_fast_tokenizer=True,)
    metrics_dict['bs_precision'] = bertscore_results['precision'][0]
    metrics_dict['bs_recall'] = bertscore_results['recall'][0]
    metrics_dict['bs_f1'] = bertscore_results['f1'][0]

    print(metrics_dict['bs_f1'])

    # bs_precision, bs_recall, bs_f1 = score([generated_summary], [gold_summary], model_type='microsoft/deberta-large-mnli', lang="en", verbose=True)


    # BERTScore better human correlation and handling of longer lengths
    #  facebook/bart-large-mnli
    # Note: this is not measuring overall faithfulness to SOURCE text as
    # is comparison to gold
    bs_precision, bs_recall, bs_f1 = score([generated_summary], [gold_summary], model_type='facebook/bart-large-mnli', lang="en", verbose=True)
    metrics_dict['bs_mnli_precision'] = bs_precision.tolist()[0]
    metrics_dict['bs_mnli_recall'] = bs_recall.tolist()[0]
    metrics_dict['bs_mnli_f1'] = bs_f1.tolist()[0]

    print(metrics_dict['bs_mnli_f1'])

    # FInal
    print(metrics_dict)

    # Return all the metrics

    return metrics_dict


def remove_nones(data):
    return [i for i in data if i is not None]


def dist_list_stats(data, title, x_label, filename):
    data = remove_nones(data)

    print('\n\n')
    print(title)
    print('Length after nones removed')
    print(len(data))
    print('MIN')
    print(min(data))
    print('MEAN')
    print(statistics.mean(data))
    print('MAX')
    print(max(data))
    graphs.histogram(data, title, x_label, 'Frequency', filename, bin_num=20, log_y=False)


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
