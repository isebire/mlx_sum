# Make a new dataset with the cleaned sources that is NER tagged
# For each of the 3 splits (3177, 454, 908 rows)
# Cols:
# id, sources_clean, source_entities, summary/long_w_chain, summary/short_w_chain, summary/tiny_w_chain

print('Running!')

from NER import ner_tag as ner
from NER import entity_stats as entity_stats
import clean
from datasets import load_dataset, Dataset, DatasetDict
import datasets
import pandas
import statistics
import lexnlp.nlp.en.segments.sentences

# Load dataset
mlx = load_dataset("allenai/multi_lexsum", name="v20220616")
mlx_semi_clean = load_dataset("isebire/multi_lexsum_CLEANED_2")

# Initialise the dataset dict that we will build by adding each split
mlx_clean = DatasetDict()

mlx_clean['train'] = mlx_semi_clean['train']


for split in ['validation', 'test']:
    print('*** Now analysing split: ' + split)

    split_data = []

    new_short_summaries_in_split = 0
    short_summaries_in_split = 0
    existing_short_summaries_in_split = 0

    # Process the data for each case
    cases_in_split = len(mlx[split])
    for i, case in enumerate(mlx[split]):
        print('** Now analysing case ' + str(i + 1) + ' of ' + str(cases_in_split) + ' in split')

        case_data = {}  # this will hold the row of the dataset for this case

        case_data['id'] = case['id']

        # Clean each source document and store in a list, like before
        print('* Cleaning source documents')
        cleaned_doc_list = [clean.clean_document(i) for i in case['sources']]
        source_text = ''
        for doc in cleaned_doc_list:
            source_text = source_text + '\n\n' + doc

        case_data['sources_clean'] = cleaned_doc_list

        for sum_len in ['long', 'short', 'tiny']:
            print('* Considering summary length: ' + sum_len)
            if case['summary/' + sum_len] is not None:
                summary = case['summary/' + sum_len]

                # Entity tags the summary
                print('* NER tagging')
                entities_w_labels, entities, labels, chain = ner.ner_legalbert_passage_by_sentence(summary)

                summary_w_chain = '[ENTITYCHAIN] ' + chain + ' [SUMMARY] ' + summary

                case_data['summary/' + sum_len + '_w_chain'] = summary_w_chain

                if sum_len == 'short':
                    existing_short_summaries_in_split += 1
                    short_summaries_in_split += 1

            else:
                case_data['summary/' + sum_len + '_w_chain'] = None

        split_data.append(case_data)

    print('** Making dataset for split')

    # Make the list of dicts into a pandas dataframe
    df = pandas.DataFrame.from_dict(split_data)

    # Make the pandas dataframe into a hugging face dataset
    ds = Dataset.from_pandas(df)

    # Add this to the overall DatasetDict
    mlx_clean[split] = ds

    # Export to csv just in case
    filename = 'mlx_clean_2' + split + '.csv'
    ds.to_csv(filename, index = None)

# output the DatasetDict huggingface
print('*** Uploading to huggingface')
mlx_clean.push_to_hub("isebire/multi_lexsum_CLEANED_AGAIN", private=True)
