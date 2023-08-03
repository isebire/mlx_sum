# Evaluation dump - assuming gold and pegasus summaries
# This is for a single one of the pegasus runs
# SET *** STUFF BELOW

# NOTE!!! gold summary entailment and gold summary readibility only calculated for 2d

print('Entered file!')

from datasets import load_dataset, load_from_disk, load_metric
import GRAPHS as graphs
import statistics
from readability import Readability
import pandas
from collections import Counter
import math
import re
import argparse
from readability_score.calculators.fleschkincaid import *

print('Imports done!')

# ***
RUN_NAME = 'eval_1cO'
FILE_NAME = 'outputs_pegasus_1c.csv'
CHAIN = False
RUN_TYPE = 1   # 1-5
INPUT_TYPE = 'c'
ANALYSE_OREO = True
ANALYSE_BERT = False

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

    ## READABILITY
    print('** Readability....')

    summary_list = [generated_summary]
    if RUN_TYPE == 2 and INPUT_TYPE == 'd':
        summary_list = [generated_summary, gold_summary]

    for text in summary_list:
        r = Readability(text)   # must contain at least 100 words

        if text == generated_summary:
            prefix = 'pegasus_'
        elif text == gold_summary:
            prefix = 'gold_'

        # need at least 100 words
        fn = prefix + 'flesch_kincaid'
        try:
            flesch_kincaid = r.flesch_kincaid().score   # .score or .grade_level
            metrics_dict[fn] = flesch_kincaid
        except:
            metrics_dict[fn] = None

        fk_2 = FleschKincaid(text).min_age
        metrics_dict['fk_2'] = fk_2

        fn = prefix + 'coleman_liau'
        try:
            coleman_liau = r.coleman_liau().score   # .score or .grade_level
            metrics_dict[fn] = coleman_liau
        except:
            metrics_dict[fn] = None

        fn = prefix + 'ari'
        try:
            ari = r.ari().score   # .score or .grade_level
            metrics_dict[fn] = ari
        except:
            metrics_dict[fn] = None

        # requires 30 sentences so may give an error
        fn = prefix + 'smog'
        try:
            smog = r.smog().score   # .score or .grade_level
            metrics_dict[fn] = smog
        except:
            metrics_dict[fn] = None

    # FInal
    print(metrics_dict)

    # Return all the metrics

    return metrics_dict


def remove_nones(data):
    return [i for i in data if not math.isnan(i)]


def dist_list_stats(data, title, x_label, filename):
    data = list(data)
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
    # graphs.histogram(data, title, x_label, 'Frequency', filename, bin_num=20, log_y=False)


# MAIN
def main():

    if ANALYSE_BERT:

        print('** Loading results csv')
        file = FILE_NAME
        results_df = pandas.read_csv(file)

        hallucinated_entity_types = []
        results = []

        for idx, case in results_df.iterrows():
            print('*** Analysing case ' + str(idx))

            gold = case['gold']
            source = case['source']
            generated = case['pegasus_output']

            gold_segmented = segmented_df.iloc[idx]['gold_sents'].split('[SENTMARKER]')
            generated_segmented = segmented_df.iloc[idx]['pegasus_output_sents'].split('[SENTMARKER]')

            # If chain or not
            metrics_dict = summary_eval(generated, generated_segmented, gold, gold_segmented, source, CHAIN, RUN_TYPE)
            results.append(metrics_dict)

        #### Aggregrate analysis
        results_df = pandas.DataFrame(results)

        # iterate over each column
        for col_name, col_data in results_df.items():
            print('** Analysing column: ' + str(col_name))
            data = col_data.values
            fn = RUN_NAME + '_dist_' + col_name
            dist_list_stats(data, str(col_name), str(col_name), fn)


    if ANALYSE_OREO:    # do it all again with oreo version
        print('**** Analysing with oreo version...')
        print('** Loading results csv')
        file = 'inputs/' + FILE_NAME
        results_df = pandas.read_csv(file)
        file = 'inputs/sents_' + FILE_NAME
        segmented_df = pandas.read_csv(file)

        hallucinated_entity_types = []
        results = []

        for idx, case in results_df.iterrows():
            print('*** Analysing case ' + str(idx))

            gold = case['gold']
            source = case['source']
            generated = case['oreo_pegasus_output']

            gold_segmented = segmented_df.iloc[idx]['gold_sents'].split('[SENTMARKER]')
            generated_segmented = segmented_df.iloc[idx]['oreo_pegasus_output_sents'].split('[SENTMARKER]')

            # If chain or not
            metrics_dict = summary_eval(generated, generated_segmented, gold, gold_segmented, source, CHAIN, RUN_TYPE)
            results.append(metrics_dict)

        #### Aggregrate analysis
        results_df = pandas.DataFrame(results)

        # iterate over each column
        for col_name, col_data in results_df.items():
            print('** Analysing column: ' + str(col_name))
            data = col_data.values
            fn = RUN_NAME + '_OREO_dist_' + col_name
            dist_list_stats(data, str(col_name), str(col_name), fn)



if __name__ == "__main__":
    main()
