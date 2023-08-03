# Comparison of metrics across all runs
# **** things to fill in

import pandas
import GRAPHS as graphs
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PegasusTokenizer
import numpy as np

# All the columns of pegasus metrics with titles ****
COLNAMES = {'r2_f1': 'ROUGE-2 F1'}

# Load csv files (do for all runs)
RUN_NAMES = ['eval_1d', 'eval_2d', 'eval_3d', 'eval_4d', 'eval_5d']
run_labels = [i.replace('_', ' ').title() for i in RUN_NAMES]

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
comparison_metric = []
for i in range(len(run_dfs[0]['r2_f1'])):
    R = (run_dfs[0]['r1_f1'][i] + run_dfs[0]['r2_f1'][i]) / 2
    comparison_metric.append(R)

pearson_correlation = np.corrcoef(source_token_counts, comparison_metric)
print(pearson_correlation[0, 1])  # it returns a matrix!
