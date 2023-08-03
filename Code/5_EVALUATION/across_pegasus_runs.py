# Comparison of metrics across all runs
# **** things to fill in

import pandas
import GRAPHS as graphs
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PegasusTokenizer
import numpy as np

# All the columns of pegasus metrics with titles ****
COLNAMES = {'r1_recall': 'Rouge 1 Recall', 'r1_precision': 'Rouge 1 Precision', ...}

# Columns where there is also a corresponding gold one ****
COLS_INCLUDE_GOLD_COMPARISON = {'pegasus_entailment': 'gold_entailment', ...}

# Load csv files (do for all runs)
RUN_NAMES = ['chain_surface', ...]  # **** etc
run_labels = [i.split('_').title() for i in RUN_NAMES]

run_dfs = []   # [df_run1, ...] etc
for run_name in RUN_NAMES:
    fn = run_name + '_metrics.csv'
    df = pandas.read_csv(fn)
    run_dfs.append(df)

# Comparing distributions across different models
for column in COLNAMES.keys():

    cols_to_compare = []  # list of lists as box plot input
    for df in run_dfs:
        cols_to_compare.append(df[column])

    labels = run_labels

    if column in COLS_INCLUDE_GOLD_COMPARISON.keys():
        gold_col = run_dfs[0][COLS_INCLUDE_GOLD_COMPARISON[column]]
        cols_to_compare.append(gold_col)
        labels.append('Gold')

    title = COLNAMES[column]
    y_label = COLNAMES[column]
    filename = column + '_comparison_dist'

    graphs.box_plot(cols_to_compare, labels, title, 'Summary', y_label, filename)


# Does summary quality correlate with number of source documents / source document length?
mlx = load_dataset("isebire/mlx_CLEANED_ORDERED_CHAINS")
mlx_test = mlx['test']
tokenizer = PegasusTokenizer.from_pretrained("nsi319/legal-pegasus")

source_token_counts = []
for i, case in enumerate(mlx_test):
    print('** ON case ' + str(i))

    source_documents = case['sources_clean']
    sources_flat = '\n[DOCSPLIT]\n'.join(source_documents)
    source_tokens = len(tokenizer.encode(sources_flat))
    source_token_counts.append(source_tokens)


# pick something to compare to (run and column)
# best option? see later. just something for now ****
comparison_metric_column = run_dfs[0]['r2_recall']
pearson_correlation = np.corrcoef(source_token_counts, comparison_metric_column)
print(pearson_correlation[0, 1])  # it returns a matrix!
