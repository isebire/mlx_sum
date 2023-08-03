# reusable code for figures

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


def pie_chart(data_list, labels_list, title, filename, wheel=True):
    # normalise all data into percentages
    data = [x * 100 / sum(data_list) for x in data_list]
    data, labels_list = zip(*sorted(zip(data, labels_list)))
    fig, ax = plt.subplots(figsize=(10,10))
    colors = plt.cm.Spectral(np.linspace(0,1,len(data_list)))
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


def box_plot(list_of_lists, group_names, title, x_label, y_label, filename):
    fig = plt.figure(figsize =(20, 20))

    # Creating axes instance
    ax = fig.add_axes([0, 0, 1, 1])

    # Creating plot
    bp = ax.boxplot(list_of_lists, patch_artist = True)

    colourmap = plt.get_cmap("Spectral")
    colors = plt.cm.Spectral(np.linspace(0,1,len(group_names)))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    for median in bp['medians']:
        median.set(color ='black',linewidth = 5)

    for flier in bp['fliers']:
        flier.set(marker ='D',color ='#e7298a',alpha = 0.5)

    ax.set_xticklabels(group_names)
    plt.title(title, fontsize=40)
    plt.xlabel(x_label, fontsize=30)
    plt.ylabel(y_label, fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    fig.savefig(filename, bbox_inches='tight')
