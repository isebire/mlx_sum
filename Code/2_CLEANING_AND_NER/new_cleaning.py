# Make a new dataset with the cleaned sources that is NER tagged
# For each of the 3 splits (3177, 454, 908 rows)
# Cols:
# id, sources_clean, source_entities, summary/long_w_chain, summary/short_w_chain, summary/tiny_w_chain

print('Running!')

import clean
from datasets import load_dataset, Dataset, DatasetDict
import datasets
import pandas
import statistics
import lexnlp.nlp.en.segments.sentences

# Load dataset
mlx = load_dataset("allenai/multi_lexsum", name="v20220616")
mlx_semi_clean = load_dataset("isebire/multi_lexsum_CLEANED_AGAIN")

# Initialise the dataset dict that we will build by adding each split
mlx_clean = DatasetDict()

for split in ['train', 'validation', 'test']:
    print('*** Now analysing split: ' + split)

    split_data = []

    for i, case in enumerate(mlx_semi_clean[split]):
        print('Cleaning case: ' + str(i))


        case_data = {}  # this will hold the row of the dataset for this case

        # Clean the cases for new sources_clean by fetching case with corresponding id in
        # original MLX
        # will always be found
        for original_case in mlx[split]:
            if original_case['id'] == case['id']:
                break

        print('* Cleaning source documents')
        cleaned_doc_list = [clean.clean_document(i) for i in original_case['sources']]
        source_text = ''
        for doc in cleaned_doc_list:
            source_text = source_text + '\n\n' + doc

        case_data['sources_clean'] = cleaned_doc_list

        # id, summary/long_w_chain, summary/short_w_chain, summary/tiny_w_chain same as prev ver
        case_data['id'] = case['id']
        case_data['summary/long_w_chain'] = case['summary/long_w_chain']
        case_data['summary/short_w_chain'] = case['summary/short_w_chain']
        case_data['summary/tiny_w_chain'] = case['summary/tiny_w_chain']

        split_data.append(case_data)

    print('** Making dataset for split')

    # Make the list of dicts into a pandas dataframe
    df = pandas.DataFrame.from_dict(split_data)

    # Make the pandas dataframe into a hugging face dataset
    ds = Dataset.from_pandas(df)

    # Add this to the overall DatasetDict
    mlx_clean[split] = ds

    # Export to csv just in case
    filename = 'mlx_clean_FINAL' + split + '.csv'
    ds.to_csv(filename, index = None)

# output the DatasetDict huggingface
print('*** Uploading to huggingface')
mlx_clean.push_to_hub("isebire/mlx_CLEAN_FINAL", private=True)
