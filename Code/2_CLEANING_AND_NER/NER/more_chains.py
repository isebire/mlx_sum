
# Cols:
# id, sources_clean, summary/short_no_chain, summary/short_chain_surface, summary/short_chain_label, summary/short_chain_both

print('Running!')

import ner_tag_2 as ner
from datasets import load_dataset, Dataset, DatasetDict
import datasets
import pandas

# Load dataset -> newest ver
mlx = load_dataset("isebire/mlx_CLEAN_FINAL")  # CHANGE

# Initialise the dataset dict that we will build by adding each split
mlx_clean = DatasetDict()

for split in ['train', 'validation', 'test']:
    print('*** Now analysing split: ' + split)

    # We are going to make a huggingface Dataset for each of these,
    # by making a pandas dataframe first

    split_data = []   # [{'sources_clean': [], etc}]  # list of dicts

    # Process the data for each case
    cases_in_split = len(mlx[split])
    for i, case in enumerate(mlx[split]):
        print('** Now analysing case ' + str(i + 1) + ' of ' + str(cases_in_split) + ' in split')

        case_data = {}  # this will hold the row of the dataset for this case

        case_data['id'] = case['id']
        case_data['sources_clean'] = case['sources_clean']

        if case['summary/short_w_chain'] is not None:
            summary_chain_surface = case['summary/short_w_chain']
            summary = summary_chain_surface.split(' [SUMMARY] ')[1]
            entities_w_labels, entities, labels, chain_surface, chain_label, chain_both = ner.ner_legalbert_passage_by_sentence(summary)

            summary_chain_label = '[ENTITYCHAIN] ' + chain_label + ' [SUMMARY] ' + summary
            summary_chain_both = '[ENTITYCHAIN] ' + chain_both + ' [SUMMARY] ' + summary

            case_data['summary/short_chain_surface'] = summary_chain_surface
            case_data['summary/short_chain_label'] = summary_chain_label
            case_data['summary/short_chain_both'] = summary_chain_both
            case_data['summary/short_no_chain'] = summary

            #print(summary_chain_both)
            #input('fish')

        else:
            print('Skipping as no short summary')

        split_data.append(case_data)


    print('** Making dataset for split')

    # Make the list of dicts into a pandas dataframe
    df = pandas.DataFrame.from_dict(split_data)

    # Make the pandas dataframe into a hugging face dataset
    ds = Dataset.from_pandas(df)

    # Add this to the overall DatasetDict
    mlx_clean[split] = ds

    # Export to csv just in case
    filename = 'mlx_notordered_3_chains_' + split + '.csv'
    ds.to_csv(filename, index = None)


# output the DatasetDict huggingface
print('*** Uploading to huggingface')
mlx_clean.push_to_hub("isebire/mlx_SHORT3CHAIN", private=True)
