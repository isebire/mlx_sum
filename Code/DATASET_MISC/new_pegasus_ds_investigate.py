import pandas
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

# Load pegasus dataset
mlx_pegasus = load_dataset('isebire/mlx_PEGASUS_WBERT')

# Load original MLX dataset
mlx_original = load_dataset("allenai/multi_lexsum", name="v20220616")

# Reproducing
#mlx_reproduce = load_dataset('isebire/mlx_og_reproducing')
#for case in mlx_reproduce['train']:
#    input(case['summary/short'])

# Setup datastructure for new dataset
mlx_new = DatasetDict()
mlx_new_train = []

cases_removed = 0
for idx, case in enumerate(mlx_pegasus['train']):
    print('*** On case ' + str(idx))

    # get the id of the case
    case_id = case['id']

    print(case_id)

    # match to case in original
    for original_idx, original_case in enumerate(mlx_original['train']):
        if original_case['id'] == case_id:
            print('Matched to original case')
            break

    if original_case['summary/short'] is None:
        # if the case doesn't have a short summary it means the current one is a long summary
            # print this long summary
        print('Short summary is really long')
        # print('* ORIGINAL')
        # print(original_case['summary/long'])
        # print('* PEGASUS')
        # print(case['summary/short_no_chain'])

        cases_removed += 1

    else:
        # otherwise - short summary is real so add to new dataset
        print('Real short summary')
        # print('* ORIGINAL')
        # print(original_case['summary/short'])
        #
        # print('* PEGASUS')
        # print(case['summary/short_no_chain'])
        mlx_new_train.append(case)

    # input('next?')


new_train_df = pandas.DataFrame.from_dict(mlx_new_train)
new_train_ds = Dataset.from_pandas(new_train_df)

print('New number of cases in train: ' + str(len(mlx_new_train)))
print('Cases removed: ' + str(cases_removed))

mlx_new['train'] = new_train_ds
# mlx_new test and val are same as mlx_pegasus
mlx_new['validation'] = mlx_pegasus['validation']
mlx_new['test'] = mlx_pegasus['test']

# upload to hf and save as .hf files
for split in ['train', 'validation', 'test']:
    filename = '230726_MLX_PEGASUS_' + split + '.hf'
    mlx_new[split].save_to_disk(filename)

print('*** Uploading to huggingface')
mlx_new.push_to_hub("isebire/230726_mlx_PEGASUS", private=True)
