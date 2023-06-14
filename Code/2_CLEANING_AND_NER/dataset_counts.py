# counting increase in number of short summaries

from datasets import load_dataset
import pandas
import pickle
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# original dataset
mlx = load_dataset("allenai/multi_lexsum", name="v20220616")

# cleaned dataset
mlx_clean = load_dataset('isebire/multi_lexsum_CLEANED_AGAIN')


print('Based from original dataset')
for split in ['train', 'validation', 'test']:
    print('Overall number of cases in ' + split)
    print(len(mlx[split]))

    existing_short = 0
    total_short = 0
    new_short = 0

    for case in mlx[split]:

        short_summary = case['summary/short']
        long_summary_words = ' '.join(case['summary/long'].replace('\n\n', ' ').split()).split(' ')
        num_long_summary_words = len(long_summary_words)

        if short_summary is not None:
            existing_short += 1
            total_short += 1

        else:
            if num_long_summary_words <= 671: # the max words in a short summary
                new_short += 1
                total_short += 1

    print('For split:')
    print('Existing short: ' + str(existing_short))
    print('New short: ' + str(new_short))
    print('Total short: ' + str(total_short))


print('Short summaries in new dataset!')
for split in ['train', 'validation', 'test']:
    print('Overall number of cases in ' + split)
    print(len(mlx_clean[split]))

    total_short = 0

    for case in mlx_clean[split]:

        short_summary = case['summary/short_w_chain']

        if short_summary is not None:
            total_short += 1

    print('For split:')
    print('Total short: ' + str(total_short))
