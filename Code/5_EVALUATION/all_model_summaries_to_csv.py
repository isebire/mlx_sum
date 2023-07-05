# Comparison of metrics across all runs

import pandas
import GRAPHS as graphs
from datasets import load_dataset, load_from_disk

# Load csv files (do for all runs)
RUN_NAMES = ['chain_surface', ...]  # etc

print('*** Loading pegasus outputs for all runs')

run_dfs = []   # [df_run1, ...] etc
for run_name in RUN_NAMES:
    fn = run_name  # idk format
    df = pandas.read_csv(fn)
    run_dfs.append(df)

print('*** Iterating over test set cases')

# Make a dataframe with summary for that case from every model

# [0] doesn't really matter, just for indexing rows
summaries_data = []
for idx, case in run_dfs[0].iterrows():
    print('* On case ' + str(idx))

    case_summaries = {}

    # Add model outputs for each model
    for i, run_name in enumerate(RUN_NAMES):

        # Get dataset -> row -> summary
        case_summaries[run_name] = run_dfs[i].iloc[idx]['pegasus_output']

    # Add gold (same in all datasets)
    case_summaries['gold'] = case['gold']

    summaries_data.append(case_summaries)

summaries_df = pandas.DataFrame(summaries_data)
summaries_df.to_csv('summaries_comparison.csv')
