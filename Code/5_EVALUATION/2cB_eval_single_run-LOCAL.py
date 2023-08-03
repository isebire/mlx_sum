# Evaluation dump - assuming gold and pegasus summaries
# This is for a single one of the pegasus runs
# SET *** STUFF BELOW

# NOTE!!! gold summary entailment and gold summary readibility only calculated for 2d

print('Entered file!')

from datasets import load_dataset, load_from_disk, load_metric
import GRAPHS as graphs
import statistics
import pandas
import ner_tag_2 as ner
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
RUN_NAME = 'chain_2cB'
FILE_NAME = 'outputs_pegasus_2cB.csv'
CHAIN = False
RUN_TYPE = 2   # 1-5
INPUT_TYPE = 'c'
ANALYSE_OREO = False
ANALYSE_BERT = True

print('*** RUN *** ')
print(RUN_NAME)

def summary_eval(generated_output, gold_output, source, chain, RUN_TYPE, hallucinated_entity_types):

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

    ## CHAIN AND ENTITY ANALYSIS

    source_upper = source.upper()

    run_chain_analysis = chain and (not malformed_chain) and (RUN_TYPE == 3)

    if run_chain_analysis:
        # Note: remember the entities in the gold chain are the same as the
        # entities in the gold summary :)
        gold_entities = gold_chain.split('|')
        gold_entities = [x.strip().strip('|').strip().upper() for x in gold_entities]
        gold_entities = [x for x in gold_entities if x != '']

        # note this would only work for type 3 (surface form chain)
        pegasus_chain_entities = generated_chain.split('|')
        pegasus_chain_entities = [x.strip().strip('|').strip().upper() for x in pegasus_chain_entities]
        pegasus_chain_entities = [x for x in pegasus_chain_entities if x != '']
        if len(pegasus_chain_entities) == 0:
            print('Skipping as no entities in chain')
            return metrics_dict, hallucinated_entity_types

    else:
        gold_entities_w_labels, gold_entities, gold_labels, _, _, _ = ner.ner_legalbert_passage_by_sentence(gold_summary)
        gold_entities = [x.upper() for x in gold_entities]

    if len(gold_entities) == 0:
        print('Skipping as no entities in gold')
        return metrics_dict, hallucinated_entity_types

    # Get entites in generated summary
    pegasus_sum_entities_w_labels, pegasus_sum_entities, pegasus_sum_labels, _, _, _ = ner.ner_legalbert_passage_by_sentence(generated_summary)
    pegasus_sum_entities = [x.upper() for x in pegasus_sum_entities]

    if len(pegasus_sum_entities) == 0:
        print('Skipping as no entities in pegasus summary')
        return metrics_dict, hallucinated_entity_types

    if run_chain_analysis:
        num_pegasus_chain_entities = len(pegasus_entities)
        num_pegasus_chain_entities_in_gold = 0
        num_pegasus_chain_entities_in_source = 0
        num_pegasus_chain_entities_in_pegasus_summary = 0

        for pegasus_chain_entity in pegasus_chain_entities:
            if pegasus_chain_entity in gold_entities:
                num_pegasus_chain_entities_in_gold += 1
            if pegasus_chain_entity in source_upper:
                num_pegasus_chain_entities_in_source += 1
            # PEGASUS: Is everything in the chain in the summary?
            if pegasus_chain_entity in pegasus_sum_entities:
                num_pegasus_chain_entities_in_pegasus_summary += 1
        pegasus_chain_entities_in_gold = num_pegasus_chain_entities_in_gold / num_pegasus_chain_entities
        metrics_dict['chain_pegasus_chain_entities_in_gold'] = pegasus_chain_entities_in_gold
        pegasus_chain_entities_in_source = num_pegasus_chain_entities_in_source / num_pegasus_chain_entities
        metrics_dict['chain_pegasus_chain_entities_in_source'] = pegasus_chain_entities_in_source
        pegasus_chain_entities_in_pegasus_summary = num_pegasus_chain_entities_in_pegasus_summary / num_pegasus_chain_entities
        metrics_dict['chain_pegasus_chain_entities_in_pegasus_summary'] = pegasus_chain_entities_in_pegasus_summary

    # Does the gold summary even correspond to the extracts?
    num_gold_entities = len(gold_entities)
    num_gold_entities_in_source = 0
    num_gold_entities_in_pegasus_sum = 0
    for gold_entity in gold_entities:
        if gold_entity in source_upper:
            num_gold_entities_in_source += 1
        if gold_entity in pegasus_sum_entities:
            num_gold_entities_in_pegasus_sum += 1

    gold_precision_source = num_gold_entities_in_source / num_gold_entities
    metrics_dict['chain_gold_precision_source'] = gold_precision_source
    recall_target = num_gold_entities_in_pegasus_sum / num_gold_entities
    metrics_dict['chain_recall_target'] = recall_target

    # PEGASUS: Is everything in the summary in the chain?
    num_pegasus_sum_entities = len(pegasus_sum_entities)
    if run_chain_analysis:
        num_pegasus_sum_entities_in_pegasus_chain = 0
    num_pegasus_sum_entities_in_source = 0
    num_pegasus_sum_entities_in_gold = 0
    for idx, pegasus_sum_entity in enumerate(pegasus_sum_entities):
        if run_chain_analysis:
            if pegasus_sum_entity in pegasus_chain_entities:
                num_pegasus_sum_entities_in_pegasus_chain += 1
        if pegasus_sum_entity in source_upper:
            num_pegasus_sum_entities_in_source += 1
        else:
            hallucinated_entity_types.append(pegasus_sum_labels[idx])
        if pegasus_sum_entity in gold_entities:
            num_pegasus_sum_entities_in_gold += 1

    if run_chain_analysis:
        pegasus_sum_entities_in_pegasus_chain = num_pegasus_sum_entities_in_pegasus_chain / num_pegasus_sum_entities
        metrics_dict['chain_pegasus_sum_entities_in_pegasus_chain'] = pegasus_sum_entities_in_pegasus_chain
    pegasus_precision_source = num_pegasus_sum_entities_in_source / num_pegasus_sum_entities
    metrics_dict['chain_pegasus_precision_source'] = pegasus_precision_source
    precision_target = num_pegasus_sum_entities_in_gold / num_pegasus_sum_entities
    metrics_dict['chain_precision_target'] = precision_target

    if precision_target + recall_target > 0:
        f1_target = (2 * precision_target * recall_target) / (precision_target + recall_target)
    else:
        f1_target = 0
    metrics_dict['chain_f1_target'] = f1_target
    if run_chain_analysis:
        if pegasus_chain_entities_in_pegasus_summary + pegasus_sum_entities_in_pegasus_chain > 0:
            f1_pegasus_chain_and_sum = (2 * pegasus_chain_entities_in_pegasus_summary * pegasus_sum_entities_in_pegasus_chain) / (pegasus_chain_entities_in_pegasus_summary + pegasus_sum_entities_in_pegasus_chain)
        else:
            f1_pegasus_chain_and_sum = 0
        metrics_dict['chain_f1_pegasus_chain_and_sum'] = f1_pegasus_chain_and_sum

    # FInal
    print(metrics_dict)

    return metrics_dict, hallucinated_entity_types


def remove_nones(data):
    return [i for i in data if not math.isnan(i)]


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
def main(args):

    if ANALYSE_BERT:

        print('** Loading results csv')
        file = f'{args.input_dir}/' + FILE_NAME
        results_df = pandas.read_csv(file)

        hallucinated_entity_types = []
        results = []

        for idx, case in results_df.iterrows():
            print('*** Analysing case ' + str(idx))

            gold = case['gold']
            source = case['bert_source']
            generated = case['pegasus_output']

            # If chain or not
            metrics_dict, hallucinated_entity_types = summary_eval(generated, gold, source, CHAIN, RUN_TYPE, hallucinated_entity_types)
            if metrics_dict != {}:
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

        hallucinated_entity_types = []
        results = []

        for idx, case in results_df.iterrows():
            print('*** Analysing case ' + str(idx))

            gold = case['gold']
            source = case['oreo_source']
            generated = case['oreo_pegasus_output']

            # If chain or not
            metrics_dict, hallucinated_entity_types = summary_eval(generated, gold, source, CHAIN, RUN_TYPE, hallucinated_entity_types)
            if metrics_dict != {}:
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
