# Evaluation dump - assuming gold and pegasus summaries
# This is for a single one of the pegasus runs
# SET *** STUFF BELOW

# NOTE!!! gold summary entailment and gold summary readibility only calculated for 2d

print('Entered file!')

from rouge_score import rouge_scorer
from datasets import load_dataset, load_from_disk, load_metric
import GRAPHS as graphs
import statistics
from bert_score import score
from readability_score.calculators.fleschkincaid import *
from readability_score.calculators.colemanliau import *
from readability_score.calculators.ari import *
from readability_score.calculators.smog import *
import language_tool_python
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
RUN_NAME = 'eval_4cB'
FILE_NAME = 'outputs_pegasus_4cB.csv'
CHAIN = True
RUN_TYPE = 4   # 1-5
INPUT_TYPE = 'c'
ANALYSE_OREO = False
ANALYSE_BERT = True

print('*** RUN *** ')
print(RUN_NAME)


# Main function

def summary_eval(generated_output, generated_segmented, gold_output, gold_segmented, source, chain, RUN_TYPE):

    # Split into chain and summary if needed
    if chain:
        malformed_chain = False
        # split by [SUMMARY] if exists
        gold_chain, gold_summary = gold_output.split('[SUMMARY]')
        gold_chain = gold_chain.strip('[ENTITYCHAIN]')

        if '[CONTENTSPLIT]' in generated_output:
            generated_chain, generated_summary = generated_output.split('[CONTENTSPLIT]')
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


    print('** Readability....')

    summary_list = [generated_summary]
    if RUN_TYPE == 2 and INPUT_TYPE == 'd':
        summary_list = [generated_summary, gold_summary]

    for text in summary_list:

        if text == generated_summary:
            prefix = 'pegasus_'
        elif text == gold_summary:
            prefix = 'gold_'

        fn = prefix + 'flesch_kincaid'
        metrics_dict[fn] = FleschKincaid(text).min_age

        fn = prefix + 'coleman_liau'
        metrics_dict[fn] = ColemanLiau(text).min_age

        fn = prefix + 'ari'
        metrics_dict[fn] = ARI(text).min_age

        fn = prefix + 'smog'
        metrics_dict[fn] = SMOG(text).min_age

    # FInal
    print(metrics_dict)

    # Return all the metrics

    return metrics_dict


def remove_nones(data):
    return [i for i in data if not math.isnan(i)]


def dist_list_stats(data, title, x_label, filename):
    data = remove_nones(data)
    data = [float(x) for x in data]

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
def main(args):

    if ANALYSE_BERT:

        print('** Loading results csv')
        file = f'{args.input_dir}/' + FILE_NAME
        results_df = pandas.read_csv(file)
        file = f'{args.input_dir}/sents_' + FILE_NAME
        segmented_df = pandas.read_csv(file)

        hallucinated_entity_types = []
        results = []

        for idx, case in results_df.iterrows():
            print('*** Analysing case ' + str(idx))

            gold = case['gold']
            source = case['bert_source']
            generated = case['pegasus_output']

            gold_segmented = segmented_df.iloc[idx]['gold_sents'].split('[SENTMARKER]')
            generated_segmented = segmented_df.iloc[idx]['pegasus_output_sents'].split('[SENTMARKER]')

            # If chain or not
            metrics_dict = summary_eval(generated, generated_segmented, gold, gold_segmented, source, CHAIN, RUN_TYPE)
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

        # Pie chart of hallucinated entity types
        # THIS IS LIKELY TO HAVE ERRORS AS USING EXACT MATCH SO DON'T REPORT DIRECTLY,
        # JUST TO INFORM INVESTIGATION
        hallucinated_entity_types_dict = dict(Counter(hallucinated_entity_types))
        print(hallucinated_entity_types_dict)

    if ANALYSE_OREO:    # do it all again with oreo version
        print('**** Analysing with oreo version...')
        print('** Loading results csv')
        file = f'{args.input_dir}/' + FILE_NAME
        results_df = pandas.read_csv(file)
        file = f'{args.input_dir}/sents_' + FILE_NAME
        segmented_df = pandas.read_csv(file)

        hallucinated_entity_types = []
        results = []

        for idx, case in results_df.iterrows():
            print('*** Analysing case ' + str(idx))

            gold = case['gold']
            source = case['oreo_source']
            generated = case['oreo_pegasus_output']

            gold_segmented = segmented_df.iloc[idx]['gold_sents'].split('[SENTMARKER]')
            generated_segmented = segmented_df.iloc[idx]['oreo_pegasus_output_sents'].split('[SENTMARKER]')

            # If chain or not
            metrics_dict = summary_eval(generated, generated_segmented, gold, gold_segmented, source, CHAIN, RUN_TYPE)
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

        # Pie chart of hallucinated entity types
        # THIS IS LIKELY TO HAVE ERRORS AS USING EXACT MATCH SO DON'T REPORT DIRECTLY,
        # JUST TO INFORM INVESTIGATION
        hallucinated_entity_types_dict = dict(Counter(hallucinated_entity_types))
        print(hallucinated_entity_types_dict)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
