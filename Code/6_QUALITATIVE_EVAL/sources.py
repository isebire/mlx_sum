# space for random investigations


from datasets import load_dataset
import pandas
import pickle

mlx = load_dataset("allenai/multi_lexsum", name="v20220616")

indices = [25, 111, 114, 144, 162, 209, 369, 459, 463, 509]
index = 0
for case in mlx['test']:
    if case['summary/short'] is None:
        continue

    if index in indices:
        print(str(index) + ' : ' + case['id'])
        docs = '[DOCSPLIT]'.join(case['sources'])
        fn = 'original_sources_' + str(index) + '.txt'
        with open(fn, 'w') as f:
            f.write(docs)

    index += 1
