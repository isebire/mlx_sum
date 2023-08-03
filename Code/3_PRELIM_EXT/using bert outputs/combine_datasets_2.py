# merging the ordered sources and entity chain datasets for one master dataset

import pandas
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

# content, summary
print('** Loading ordered validation dataset HF...')
mlx_extracts = load_dataset('isebire/mlx_EXTRACTS')

# id, sources_clean, summary/short_no_chain, summary/short_chain_surface
# summary/short_chain_label, summary/short_chain_both
print('** Loading entity chain dataset....')
mlx_ordered_chains = load_dataset("isebire/mlx_CLEANED_ORDERED_CHAINS")

pegasus_ds = DatasetDict()

splits = ['train', 'validation', 'test']

for split in splits:
    split_data = []
    for i, case in enumerate(mlx_ordered_chains[split]):
        case_data = {}
        print('** Case ' + str(i) + ' of ' + split)

        case_data['id'] = case['id']
        case_data['summary/short_no_chain'] = case['summary/short_no_chain']
        case_data['summary/short_chain_surface'] = case['summary/short_chain_surface']
        case_data['summary/short_chain_label'] = case['summary/short_chain_label']
        case_data['summary/short_chain_both'] = case['summary/short_chain_both']

        case_data['extracts'] = mlx_extracts[split][i]['content']

        # print('Tests to ensure correspondance')
        # print(case['summary/short_no_chain'])
        # print(mlx_extracts[split][i]['summary'])

        split_data.append(case_data)

    # SAVE SPLIT
    # Make the list of dicts into a pandas dataframe
    df = pandas.DataFrame.from_dict(split_data)

    # Make the pandas dataframe into a hugging face dataset
    ds = Dataset.from_pandas(df)
    print(len(ds))

    pegasus_ds[split] = ds

    # Export to disk
    filename = 'mlx_pegasus_' + split + '.hf'
    ds.save_to_disk(filename)


# output the DatasetDict huggingface
print('*** Uploading to huggingface')
pegasus_ds.push_to_hub("isebire/mlx_PEGASUS", private=True)
