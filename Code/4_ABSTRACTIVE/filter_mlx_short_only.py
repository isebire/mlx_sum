# Filter MLX to only cases that have a short summary to prevent issues later

from datasets import load_dataset, Dataset
import pandas

mlx_clean = load_dataset('isebire/mlx_CLEAN_FINAL')

for split in ['train', 'validation', 'test']:

    split_data = []

    for i, case in enumerate(mlx_clean[split]):
        print('Case ' + str(i))

        if case['summary/short_w_chain'] is None:
            continue

        split_data.append(case)

    print('Cases with short summary in split')
    print(len(split_data))

    # Make the list of dicts into a pandas dataframe
    df = pandas.DataFrame.from_dict(split_data)

    # Make the pandas dataframe into a hugging face dataset
    ds = Dataset.from_pandas(df)

    # Export to csv
    filename = 'mlx-final-' + split + '.csv'
    ds.to_csv(filename)
