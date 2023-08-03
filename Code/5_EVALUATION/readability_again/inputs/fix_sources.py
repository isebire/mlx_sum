import pandas
from datasets import load_dataset

datasets = [('outputs_pegasus_1d.csv', 'd'), ('outputs_pegasus_1c.csv', 'c'),
            ('outputs_pegasus_2d.csv', 'd'), ('outputs_pegasus_2cO.csv', 'c'),
            ('outputs_pegasus_3d.csv', 'd'), ('outputs_pegasus_3cO.csv', 'c'),
            ('outputs_pegasus_4d.csv', 'd'), ('outputs_pegasus_4cO.csv', 'c'),
            ('outputs_pegasus_5d.csv', 'd'), ('outputs_pegasus_5cO.csv', 'c')]
# 2cB - 5cB are already in right format


# load MLX pegasus (input for )
mlx_pegasus = load_dataset('isebire/230726_mlx_PEGASUS')


for dataset_fn, type in datasets:
    # Read csv
    file_to_read = 'OLD_' + dataset_fn
    old_ds = pandas.read_csv(file_to_read)

    new_ds = []

    # Loop through
    for i, case in old_ds.iterrows():
        case_data = {}

        # keep columns gold, oreo_pegasus_output, pegasus_output
        case_data['gold'] = case['gold']
        case_data['oreo_pegasus_output'] = case['oreo_pegasus_output']
        try:  # c０型なら存在しないかも
            case_data['pegasus_output'] = case['pegasus_output']
        except:
            pass

        # remove source and replace with oreo_source and bert_source
        if type == 'c':
            case_data['bert_source'] = mlx_pegasus['test'][i]['bert_sents']
            case_data['oreo_source'] = mlx_pegasus['test'][i]['oreo_sents']
        elif type == 'd':
            case_data['bert_source'] = mlx_pegasus['test'][i]['bert_windows']
            case_data['oreo_source'] = mlx_pegasus['test'][i]['oreo_windows']

        new_ds.append(case_data)

    #
    new_df = pandas.DataFrame.from_dict(new_ds)
    new_df.to_csv(dataset_fn)
