# Comparison of metrics across all runs
# **** things to fill in

import pandas
import GRAPHS as graphs
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PegasusTokenizer
import numpy as np


# Load csv files (do for all runs)
RUN_NAMES = ['eval_reprod']
run_labels = [i.replace('_', ' ').title() for i in RUN_NAMES]

run_dfs = []   # [df_run1, ...] etc
for run_name in RUN_NAMES:
    fn = run_name + '_metrics.csv'
    df = pandas.read_csv(fn)
    run_dfs.append(df)

# Does summary quality correlate with number of source documents / source document length?
mlx = load_dataset("allenai/multi_lexsum", name="v20220616")
mlx_test = mlx['test']
tokenizer = PegasusTokenizer.from_pretrained("nsi319/legal-pegasus")

source_token_counts = []
counter = 0
for i, case in enumerate(mlx_test):
    if case['summary/short'] is None:
        continue

    print('** ON case ' + str(counter))

    source_documents = case['sources']
    sources_flat = '\n[DOCSPLIT]\n'.join(source_documents)
    source_tokens = len(tokenizer.encode(sources_flat))
    source_token_counts.append(source_tokens)

    counter += 1


# pick something to compare to (run and column)
# best option? see later. just something for now ****
comparison_metric = []
for i in range(len(run_dfs[0]['r2_f1'])):
    R = (run_dfs[0]['r1_f1'][i] + run_dfs[0]['r2_f1'][i]) / 2
    comparison_metric.append(R)

pearson_correlation = np.corrcoef(source_token_counts, comparison_metric)
print(pearson_correlation[0, 1])  # it returns a matrix!
