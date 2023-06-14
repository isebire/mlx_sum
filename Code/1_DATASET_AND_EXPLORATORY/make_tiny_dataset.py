# tiny dataset

from datasets import load_dataset, Dataset, DatasetDict
import datasets
import pandas

mlx = load_dataset("allenai/multi_lexsum", name="v20220616")

# Initialise the dataset dict that we will build by adding each split
mlx_clean = DatasetDict()

for split in ['train', 'validation', 'test']:
    # We are going to make a huggingface Dataset for each of these,
    # by making a pandas dataframe first

    split_data = []

    # First 5 cases for each split
    for case in [mlx[split][i] for i in range(10)]:

        print(case)

        case_data = {}  # this will hold the row of the dataset for this case

        case_data['id'] = case['id']
        case_data['sources'] = case['sources']

        for sum_len in ['long', 'short', 'tiny']:
            case_data['summary/' + sum_len] = case['summary/' + sum_len]

        split_data.append(case_data)

    # Make the list of dicts into a pandas dataframe
    df = pandas.DataFrame.from_dict(split_data)

    # Make the pandas dataframe into a hugging face dataset
    ds = Dataset.from_pandas(df)

    # Add this to the overall DatasetDict
    mlx_clean[split] = ds

    # Export to csv just in case
    filename = 'mlx_tiny_' + split + '.csv'
    ds.to_csv(filename, index = None)

# output the DatasetDict huggingface
mlx_clean.push_to_hub("isebire/multi_lexsum_TINY", private=True)
