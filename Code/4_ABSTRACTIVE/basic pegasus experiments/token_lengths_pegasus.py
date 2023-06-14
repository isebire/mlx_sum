# tokens in summary lengths

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PegasusTokenizer
import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

sns.set('talk') # alternatively, poster <- presets for font size
sns.set_style('ticks')

def histogram(list, title, x_label, y_label, filename, bin_num=20, log_y=True):
    fig, ax = plt.subplots(figsize=(20,20))
    cm = mpl.colormaps['plasma']    #'RdYlBu_r'
    n, bins, patches = plt.hist(list, edgecolor='black', bins=bin_num)
    plt.ticklabel_format(style='plain', axis='x')


    # For colouring
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))

    if log_y:
        ax.set_yscale('log')

    ax.get_yaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    if ax.get_xlim()[1] > 1000:
        ax.get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.title(title, fontsize=40)
    plt.xlabel(x_label, fontsize=30)
    plt.ylabel(y_label, fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    fig.savefig(filename, bbox_inches='tight')

mlx = load_dataset('isebire/multi_lexsum_CLEANED_AGAIN')
# tokenizer = AutoTokenizer.from_pretrained("nsi319/legal-pegasus")  # same results for AutoTokenizer and PegasusTokenizer
tokenizer = PegasusTokenizer.from_pretrained("nsi319/legal-pegasus")
model = AutoModelForSeq2SeqLM.from_pretrained("nsi319/legal-pegasus")

short_summary_tokens_no_chain = []
short_summary_tokens_chain = []

for split in ['train', 'validation', 'test']:
    print('On split ' + split)

    for case in mlx[split]:
        print('New case!! ')

        short_summary_chain = case['summary/short_w_chain']

        if short_summary_chain is None:
            continue

        short_summary_no_chain = case['summary/short_w_chain'].split('[SUMMARY]')[1].strip()

        # note: 'maximum length of input sequence is 1024 tokens'
        no_chain_tokens = tokenizer.encode(short_summary_no_chain)
        chain_tokens = tokenizer.encode(short_summary_chain)

        short_summary_tokens_no_chain.append(len(no_chain_tokens))
        short_summary_tokens_chain.append(len(chain_tokens))

print('ALL THIS IS FOR SHORT SUMMARIES AND PEGASUS')
print('With entity chain')
print('Max tokens')
print(max(short_summary_tokens_chain))
print('Min tokens')
print(min(short_summary_tokens_chain))
print('Without entity chain')
print('Max tokens')
print(max(short_summary_tokens_no_chain))
print('Min tokens')
print(min(short_summary_tokens_no_chain))

# Draw histograms
histogram(short_summary_tokens_chain, 'Number of Tokens in Short Summary (Entity Chain)', 'Number of Tokens', 'Frequency', 'short_chain_pegasus_tokens', log_y=False)
histogram(short_summary_tokens_no_chain, 'Number of Tokens in Short Summary (No Entity Chain)', 'Number of Tokens', 'Frequency', 'short_no_chain_pegasus_tokens', log_y=False)
