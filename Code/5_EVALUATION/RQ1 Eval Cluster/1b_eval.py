# Evaluation dump - assuming gold and pegasus summaries
# This is for a single one of the pegasus runs

print('Entered file!')

from rouge_score import rouge_scorer
from datasets import load_dataset, load_from_disk, load_metric
import statistics
from bert_score import score
import pandas
from collections import Counter
import math
import re
import argparse

print('Imports done!')

# Need all of this to deal with the filesystem
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True, help="Folder for all the input stuff")
parser.add_argument("--output_dir", type=str, required=True, help="Folder for all the output stuff")

# ***
RUN_NAME = 'eval_1b'
FILE_NAME = 'outputs_pegasus_1b.csv'
CHAIN = False
RUN_TYPE = 1   # 1-5
OREO_ALSO = False

print('*** RUN *** ')
print(RUN_NAME)

print('** Loading eval utils...')
scorer_rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
bertscore = load_metric("bertscore")

# Main function

def summary_eval(generated_output, gold_output, source, chain, RUN_TYPE):

    generated_summary = generated_output
    gold_summary = gold_output

    metrics_dict = {}

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

    # BERTScore for comparison with original mlx paper
    print('** BERTScore...')

    bertscore_results = bertscore.compute(predictions=[generated_summary], references=[gold_summary], lang="en", model_type="microsoft/deberta-large-mnli", rescale_with_baseline=True, use_fast_tokenizer=True,)
    metrics_dict['bs_precision'] = bertscore_results['precision'][0]
    metrics_dict['bs_recall'] = bertscore_results['recall'][0]
    metrics_dict['bs_f1'] = bertscore_results['f1'][0]

    # bs_precision, bs_recall, bs_f1 = score([generated_summary], [gold_summary], model_type='microsoft/deberta-large-mnli', lang="en", verbose=True)

    # BERTScore better human correlation and handling of longer lengths
    #  facebook/bart-large-mnli
    # Note: this is not measuring overall faithfulness to SOURCE text as
    # is comparison to gold
    bs_precision, bs_recall, bs_f1 = score([generated_summary], [gold_summary], model_type='facebook/bart-large-mnli', lang="en", verbose=True)
    metrics_dict['bs_mnli_precision'] = bs_precision.tolist()[0]
    metrics_dict['bs_mnli_recall'] = bs_recall.tolist()[0]
    metrics_dict['bs_mnli_f1'] = bs_f1.tolist()[0]

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

# MAIN


#### Actual eval metrics - do for each case and save to some data structure

def main(args):

    print('** Loading results csv')
    file = f'{args.input_dir}/' + FILE_NAME
    results_df = pandas.read_csv(file)

    results = []

    for idx, case in results_df.iterrows():
        print('*** Analysing case ' + str(idx))

        gold = case['gold']
        source = case['source']
        generated = case['pegasus_output']

        # If chain or not
        metrics_dict = summary_eval(generated, gold, source, CHAIN, RUN_TYPE)
        results.append(metrics_dict)

    #### Aggregrate analysis
    results_df = pandas.DataFrame(results)
    fn = f'{args.output_dir}/' + RUN_NAME + '_metrics.csv'
    results_df.to_csv(fn)

    # iterate over each column
    for col_name, col_data in results_df.items():
        print('** Analysing column: ' + str(col_name))
        data = col_data.values
        fn = f"{args.output_dir}/" + RUN_NAME + '_dist_' + col_name
        dist_list_stats(data, str(col_name), str(col_name), fn)

    if OREO_ALSO:    # do it all again with oreo version
        print('**** Analysing with oreo version...')
        results = []

        for idx, case in results_df.iterrows():
            print('*** Analysing case ' + str(idx))

            gold = case['gold']
            source = case['source']
            generated = case['oreo_pegasus_output']

            # If chain or not
            metrics_dict = summary_eval(generated, gold, source, CHAIN, RUN_TYPE)
            results.append(metrics_dict)

        #### Aggregrate analysis
        results_df = pandas.DataFrame(results)
        fn = f'{args.output_dir}/' + RUN_NAME + '_OREO_metrics.csv'
        results_df.to_csv(fn)

        # iterate over each column
        for col_name, col_data in results_df.items():
            print('** Analysing column: ' + str(col_name))
            data = col_data.values
            fn = f"{args.output_dir}/" + RUN_NAME + '_OREO_dist_' + col_name
            dist_list_stats(data, str(col_name), str(col_name), fn)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
