# space for random investigations


from datasets import load_dataset
import pandas
import pickle

multi_lexsum = load_dataset("allenai/multi_lexsum", name="v20220616")

for split in ['train']:

    for case in [multi_lexsum[split][0]]:  # just the classic case

        print('source documents for first validation case')

        docs = case['sources']
        for i, doc in enumerate(docs):
            print('**** Document ' + str(i + 1))
            print(doc)
            fn = 'train0_' + str(i) + '.txt'
            with open(fn, 'w') as f:
                f.write(doc)
