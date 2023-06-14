# Calculating (exploratory) stats re entities

from collections import Counter
import datetime
from dateutil import parser
import re
import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

sns.set('talk') # alternatively, poster <- presets for font size
sns.set_style('ticks')

# just copied from graphs file

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

# Proportion of entities of each type among all the entities

def entity_types(entity_type_list, filename):

    # From the list, get a count for each type
    counts_dict = entity_type_list
    pie_chart(counts_dict.values(), counts_dict.keys(), 'Entity Proportions', filename, wheel=True)


# Proportion of entities among all words (entity density)
def entity_density(entities, full_text):
    full_text_words = len(full_text.split(' '))
    entity_words = 0
    for entity in entities:
        entity_words += len(entity.split(' '))

    return entity_words / full_text_words

# Proportion of entities in summary that are in source (are there any (true) hallucinations?)

def entity_match(source_text, summary_entity, entity_type, verbose=False):

    # if summary entity is a generalisation of source summary, match

    if entity_type != 'DATE':

        # Covers eg first or last names only
        # remember casing is a thing!!!
        if summary_entity.upper() in source_text.upper():
            return True

    else:
        # This is an APPROXIMATION
        date_string = summary_entity

        date_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '/']
        # Elif Month Year format, search in text. if not found, datetime, then convert to for MM/XX/YYYY and XX/MM/YYYY. then regex
        # Elif if early / late YEAR
        if date_string.upper() in source_text.upper():
            return True

        # Expressions like 'the next day', cannot determine easily
        elif all(x not in date_chars for x in date_string):
            # Optimistic
            # print('Assuming yes for relative expressions ' + date_string)
            return True

        # If the date string contains words
        elif True in (x not in date_chars for x in date_string):

            try:  # Assuming case eg March 2003
                date = parser.parse(date_string, dayfirst=True)

                # is the date incomplete?
                # 2 options in case by chance it is truly one of these dates
                jan_01_2001 = datetime.datetime.strptime('01/01/01', '%m/%d/%y')
                feb_02_2002 = datetime.datetime.strptime('02/02/02', '%m/%d/%y')

                # Both ways and with and without 0 padding -> permissive
                date_1 = date.strftime('%m/%d/%Y')
                date_3 = date.strftime('%-m/%-d/%Y')
                date_5 = date.strftime('%-m/%-d/%y')

                date_2 = date.strftime('%d/%m/%Y')
                date_6 = date.strftime('%-d/%-m/%y')
                date_4 = date.strftime('%-d/%-m/%Y')

                month_first = [date_1, date_3, date_5]
                day_first = [date_2, date_4, date_6]


                if parser.parse(date_string, default=jan_01_2001) != parser.parse(date_string, default=feb_02_2002):
                    # date is incomplete
                    found = False

                    for date_format in month_first:
                        aspects = date_format.split['/']
                        if re.search(r'' + re.escape(aspects[0]) + '/[0-9][0-9]/' + re.escape(aspects[2]), source_text):
                            found = True
                            break

                    for date_format in day_first:
                        aspects = date_format.split['/']
                        if re.search(r'[0-9][0-9]/' + re.escape(aspects[1]) + '/' + re.escape(aspects[2]), source_text):
                            found = True
                            break

                    return found

                    if re.search(r'[0-9][0-9]' + re.escape(date_2[2:]), source_text):
                        return True

                else:
                    # date is complete
                    if date_1 in source_text or date_2 in source_text or date_3 in source_text or date_4 in source_text or date_5 in source_text or date_6 in source_text:
                        return True


            except:
                # Strip and try search for just date part
                date_part = ''
                for char in date_string:
                    if char in date_chars:
                        date_part = date_part + char

                if date_part in source_text:
                    return True

    if verbose:
        print('Could not match this entity with source: ' + summary_entity)

    return False

def entities_verified(source_text, summary_entities_w_labels):
    # measures (real) hallucination

    matches = 0
    for summary_entity, entity_type in summary_entities_w_labels:
        if entity_match(source_text, summary_entity, entity_type):
            matches += 1

    percentage_verified = matches / len(summary_entities_w_labels)
    return percentage_verified
