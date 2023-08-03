# Segmenting bc lexnlp doesn't work on fluidstack

print('Entered file!')

import pandas
import lexnlp.nlp.en.segments.sentences
import re

print('Imports done!')

def segment(text):
    docs = text.split('\n[DOCSPLIT]\n')
    sentences = []
    for doc in docs:
        if 'CIVIL DOCKET' in doc:
            sentences = sentences + doc.split('\n')
        else:
            # dealing with some common errors
            doc = doc.replace('; (', '; . (')
            doc2 = doc.replace('; WHER', '; . WHER')
            doc = doc2.replace('; and WHER', '; . and WHER')
            sentences = sentences + lexnlp.nlp.en.segments.sentences.get_sentence_list(doc)

    # removing common issues in segmentation
    new_sentences = []

    for sent in sentences:
        sent = sent.replace('\n', ' ')

        # strip out just numbers and junk (prepending paras)
        if re.search('[a-zA-Z]', sent) and len(sent.split(' ')) <= 500:
            # fixing common segmentation errors
            # merge with next if previous sentence ends v.
            if new_sentences != []:

                if new_sentences[-1].strip().endswith('v.'):
                    new_sentences[-1] = new_sentences[-1] + ' ' + sent

                # merge with previous if current starts Section
                if sent.startswith('Section'):
                    new_sentences[-1] = new_sentences[-1] + ' ' + sent

                else:
                    new_sentences.append(sent)

            else:
                new_sentences.append(sent)

    return new_sentences

file_names = ['outputs_pegasus_2cB.csv', 'outputs_pegasus_3cB.csv', 'outputs_pegasus_4cB.csv', 'outputs_pegasus_5cB.csv']
chains = [False, True, True, True]

for file_name, chain in zip(file_names, chains):
    print('For file: ')
    print(file_name)
    # read the csv
    results_df = pandas.read_csv(file_name)

    # set up df
    segmented_data = []

    for idx, case in results_df.iterrows():

        # segment summaries
        sents_data = {}
        gold_output = case['gold']
        if chain:
            gold_chain, gold_summary = gold_output.split('[SUMMARY]')
        else:
            gold_summary = gold_output
        sents_data['gold_sents'] = '[SENTMARKER]'.join(segment(gold_summary))

        pegasus_output = case['pegasus_output']
        if chain and '[CONTENTSPLIT]' in pegasus_output:
            pegasus_chain, pegasus_summary = pegasus_output.split('[CONTENTSPLIT]')
        else:
            pegasus_summary = pegasus_output
        sents_data['pegasus_output_sents'] = '[SENTMARKER]'.join(segment(pegasus_summary))

        segmented_data.append(sents_data)

    segmented_df = pandas.DataFrame(segmented_data)
    fn = 'sents_' + file_name
    segmented_df.to_csv(fn)
