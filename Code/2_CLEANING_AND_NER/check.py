# check alignment -> all good


from datasets import load_dataset
import pandas
import pickle

multi_lexsum = load_dataset('isebire/mlx_CLEAN_FINAL')

for split in ['train']:

    for case in [multi_lexsum[split][-1]]:  # just the classic case

        docs = case['sources_clean']
        for i, doc in enumerate(docs):
            print('**** Document ' + str(i + 1))
            print(doc)

        print(case['summary/short_w_chain'])
