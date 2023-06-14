# Exploratory analysis on case metadata


from datasets import load_dataset
import pandas
import pickle
import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

sns.set('talk') # alternatively, poster <- presets for font size
sns.set_style('ticks')

mlx2 = load_dataset("allenai/multi_lexsum", name="v20230518")

def pie_chart(data_list, labels_list, title, filename, wheel=True):
    # normalise all data into percentages
    data = [x * 100 / sum(data_list) for x in data_list]
    data, labels_list = zip(*sorted(zip(data, labels_list)))
    fig, ax = plt.subplots(figsize=(10,10))
    colors = plt.cm.Spectral(np.linspace(0,1,len(data_list)))   # Spectral
    ax.pie(data, colors=colors, startangle=90, wedgeprops = {"edgecolor" : "black", 'linewidth': 1})
    plt.title(title, fontsize=25)

    if wheel:
        centre_circle = plt.Circle((0,0),0.70,fc='white',color='black', linewidth=1)
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)

    labels = [f'{l}, {s:0.1f}%' for l, s in zip(labels_list, data)]
    ax.legend(loc ="center left", bbox_to_anchor=(1, 0.5), labels=labels, labelspacing=-2.0, frameon=False)
    fig.savefig(filename, bbox_inches='tight')


def bar_chart(data_dict, x_label, y_label, title, filename, log_y=True):
    data = list(data_dict.keys())
    labels = list(data_dict.values())
    fig, ax = plt.subplots(figsize=(20,20))
    colourmap = plt.get_cmap("Spectral")
    colors = plt.cm.Spectral(np.linspace(0,1,len(data)))
    plt.bar(data, labels, color=colors)
    plt.title(title, fontsize=40)
    plt.xlabel(x_label, fontsize=30)
    plt.ylabel(y_label, fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    ax.set_xticks(ax.get_xticks()[9::10])

    if log_y:
        ax.set_yscale('log')

    fig.savefig(filename, bbox_inches='tight')


case_types_dict = dict() # pass .values(), .keys() to pie
filing_year_dict = dict()

for split in ['test', 'validation', 'train']:

    print('On split ' + split)

    for case in mlx2[split]:

        case_type = case['case_metadata']['case_type']
        filing_year = case['case_metadata']['filing_year']

        filing_year_dict[filing_year] = filing_year_dict.get(filing_year, 0) + 1
        case_types_dict[case_type] = case_types_dict.get(case_type, 0) + 1


# Output stats
print(case_types_dict)
print(filing_year_dict)

# Fill in any gaps
filing_years = filing_year_dict.keys()
years_numbers = [int(x) for x in filing_years]
filing_year_dict_new = {}
for i in range(min(years_numbers), max(years_numbers) + 1):
    filing_year_dict_new[str(i)] = filing_year_dict.get(str(i), 0)

print(filing_year_dict_new)

# Make the figures
pie_chart(case_types_dict.values(), case_types_dict.keys(), 'Case Types', 'case_types')
bar_chart(filing_year_dict_new, 'Filing Year', 'Number of Cases', 'Filing Year of Cases', 'filing_year')
