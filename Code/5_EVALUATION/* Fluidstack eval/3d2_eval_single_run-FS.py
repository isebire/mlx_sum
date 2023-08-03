# Evaluation dump - assuming gold and pegasus summaries
# This is for a single one of the pegasus runs
# SET *** STUFF BELOW

# NOTE!!! gold summary entailment and gold summary readibility only calculated for 2d

print('Entered file!')

from rouge_score import rouge_scorer
from datasets import load_dataset, load_from_disk, load_metric
import GRAPHS as graphs
import statistics
from bert_score import score
from faithfulness.Entailment import Entailment, EntailmentMethod   # NOTE: using custom edited ver, not pkg!
from readability import Readability
import language_tool_python
import pandas
from collections import Counter
import math
import re
import argparse

print('Imports done!')

# Need all of this to deal with the filesystem
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True, help="Folder for all the input stuff")
parser.add_argument("--output_dir", type=str, required=True, help="Folder for all the output stuff")

# ***
RUN_NAME = 'eval_3d2'
FILE_NAME = 'outputs_pegasus_3d.csv'
CHAIN = True
RUN_TYPE = 3   # 1-5
INPUT_TYPE = 'd'
ANALYSE_OREO = True
ANALYSE_BERT = False

print('*** RUN *** ')
print(RUN_NAME)

print('** Loading eval utils...')
scorer_rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
tool = language_tool_python.LanguageTool('en-US')
entailment_metric = Entailment(method=EntailmentMethod.DOC)
bertscore = load_metric("bertscore")   # move

# helper functions

def nid(text):
    # NID = 1 -   entropy(document) / log(len(document))
    # Diversity is defined
    # as the entropy of unigrams in the document (Feigenblat et al., 2017). Since longer documents are more
    # likely to have a higher entropy, we normalize the
    # diversity with the maximum possible entropy for
    # the document log(|D|)
    #  higher NID indicates more redundancy.

    words = text.split(' ')
    word_counts = Counter(words)
    frequencies = ((i / len(words)) for i in word_counts.values())
    entropy = - sum(f * math.log(f, 2) for f in frequencies)

    nid = 1 - (entropy / math.log(len(words)))
    return nid

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

# Main function

def summary_eval(generated_output, generated_segmented, gold_output, gold_segmented, source, chain, RUN_TYPE):

    # Split into chain and summary if needed
    if chain:
        malformed_chain = False
        # split by [SUMMARY] if exists
        gold_chain, gold_summary = gold_output.split('[SUMMARY]')
        gold_chain = gold_chain.strip('[ENTITYCHAIN]')

        if '[CONTENTSPLIT]' in generated_output:
            generated_chain, generated_summary = generated_output.split('[CONTENTSPLIT]')
            generated_chain = generated_chain.strip('[ENTITYCHAIN]')
        else:
            generated_summary = generated_output
            malformed_chain = True
    else:
        generated_summary = generated_output
        gold_summary = gold_output

    metrics_dict = {}

    if chain:
        if malformed_chain:
            metrics_dict['malformed_chain'] = 1
        else:
            metrics_dict['malformed_chain'] = 0

    # Record of all columns - starred is reported in OG
    # malformed_chain (if chain one)
    # r1_precision, r1_recall, *r1_f1
    # r2_precision, r2_recall, *r2_f1
    # rL_precision, rL_recall, *rL_f1
    # bs_precision, bs_recall, *bs_f1
    # bs_mnli_precision, bs_mnli_recall, bs_mnli_f1
    # unique_trigram_ratio
    # nid
    # redundancy
    # grammatical_errors
    # pegasus_entailment, gold_entailment
    # pegasus_flesch_kincaid, pegasus_coleman_liau, pegasus_ari, pegasus_smog
    # gold_flesch_kincaid, gold_coleman_liau, gold_ari, gold_smog

    ## SUMMARY EFFECTIVENESS

    # ROUGE based (summary)
    print('** ROUGE...')
    r1_precision, r1_recall, r1_f1 = scorer_rouge.score(gold_summary, generated_summary)['rouge1']
    metrics_dict['r1_precision'] = r1_precision
    metrics_dict['r1_recall'] = r1_recall
    metrics_dict['r1_f1'] = r1_f1

    r2_precision, r2_recall, r2_f1 = scorer_rouge.score(gold_summary, generated_summary)['rouge2']
    metrics_dict['r2_precision'] = r2_precision
    metrics_dict['r2_recall'] = r2_recall
    metrics_dict['r2_f1'] = r2_f1

    rL_precision, rL_recall, rL_f1 = scorer_rouge.score(gold_summary, generated_summary)['rougeL']
    metrics_dict['rL_precision'] = rL_precision
    metrics_dict['rL_recall'] = rL_recall
    metrics_dict['rL_f1'] = rL_f1

    # BERTScore for comparison with original mlx paper
    print('** BERTScore...')

    # bertscore_results = bertscore.compute(predictions=[generated_summary], references=[gold_summary], lang="en", model_type="microsoft/deberta-large-mnli", rescale_with_baseline=True, use_fast_tokenizer=True,)
    # metrics_dict['bs_precision'] = bertscore_results['precision'][0]
    # metrics_dict['bs_recall'] = bertscore_results['recall'][0]
    # metrics_dict['bs_f1'] = bertscore_results['f1'][0]

    # This is equivalent to above but way faster
    bs_precision, bs_recall, bs_f1 = score([generated_summary], [gold_summary], model_type='microsoft/deberta-large-mnli', lang="en", rescale_with_baseline=True, use_fast_tokenizer=True, device='cuda')
    metrics_dict['bs_precision'] = bs_precision.tolist()[0]
    metrics_dict['bs_recall'] = bs_recall.tolist()[0]
    metrics_dict['bs_f1'] = bs_f1.tolist()[0]

    # BERTScore better human correlation and handling of longer lengths
    #  facebook/bart-large-mnli
    # Note: this is not measuring overall faithfulness to SOURCE text as
    # is comparison to gold
    bs_precision, bs_recall, bs_f1 = score([generated_summary], [gold_summary], model_type='facebook/bart-large-mnli', lang="en", verbose=True, device='cuda')
    metrics_dict['bs_mnli_precision'] = bs_precision.tolist()[0]
    metrics_dict['bs_mnli_recall'] = bs_recall.tolist()[0]
    metrics_dict['bs_mnli_f1'] = bs_f1.tolist()[0]

    # !! Unique ngram ratio  - trigrams? n = 1,2,3 in paper with defs
    # not trigrams as constrained when decoding
    # count (unique n grams ) / count (n grams)
    print('** Unique bigram...')
    words = generated_summary.split(' ')
    all_bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
    num_total_bigrams = len(all_bigrams)
    num_unique_bigrams = len(list(set(all_bigrams)))
    metrics_dict['unique_bigram_ratio'] = num_unique_bigrams / num_total_bigrams


    # Normalised inverse of diversity
    print('** Normalised inverse of diversity...')
    # NID = 1 -   entropy(document) / log(len(document))
    metrics_dict['nid'] = nid(generated_summary)

    # More than sentiments redundancy
    # removed bc too much trouble than it was worth
    # original: df['cleaned_data'] = df.text.apply(mts.clean_data, args=(True, True, False, True, False))
    # mts_cleaned_text = mts.clean_data(generated_summary, True, True, False, True, False)
    # redundancy = mts.Redundancy(mts_cleaned_text, n = 10)
    # metrics_dict['redundancy'] = redundancy

    # Grammaticality (language tool)
    print('** Grammaticality...')
    errors = tool.check(generated_summary)
    # It can also correct = tool.correct(text)
    metrics_dict['grammatical_errors'] = len(errors)

    ## ENTAILMENT (faithfulness library)

    print('** Entailment...')

    # facebook/bart-large-mnli  (same as new BERTScore)
    # it is already less than 1024 tokens as 'source' as what is pegasus input
    # so do not need chunking

    # entailment_metric = Entailment(method=EntailmentMethod.DOC)   # moved

    # segement summary into sentences
    generated_summary_sentences = generated_segmented
    entailments = []

    for generated_summary_sentence in generated_summary_sentences:
        entailment_result = entailment_metric.score(generated_summary_sentence, source)
        entailments.append(entailment_result)

    # Mean across all sents in the summary
    metrics_dict['pegasus_entailment'] = statistics.mean(entailments)

    # also do for gold summaries!! -> only need to do once so just do for run type 2

    if RUN_TYPE == 2 and INPUT_TYPE == 'd':

        gold_summary_sentences = gold_segmented
        entailments = []

        for gold_summary_sentence in gold_summary_sentences:
            entailment_result = entailment_metric.score(gold_summary_sentence, source)
            entailments.append(entailment_result)

        # Mean across all sents in the summary
        metrics_dict['gold_entailment'] = statistics.mean(entailments)

    ## READABILITY
    print('** Readability....')

    summary_list = [generated_summary]
    if RUN_TYPE == 2 and INPUT_TYPE == 'd':
        summary_list = [generated_summary, gold_summary]

    for text in summary_list:
        r = Readability(text)   # must contain at least 100 words

        if text == generated_summary:
            prefix = 'pegasus_'
        elif text == gold_summary:
            prefix = 'gold_'

        # need at least 100 words
        fn = prefix + 'flesch_kincaid'
        try:
            flesch_kincaid = r.flesch_kincaid().score   # .score or .grade_level
            metrics_dict[fn] = flesch_kincaid
        except:
            metrics_dict[fn] = None

        fn = prefix + 'coleman_liau'
        try:
            coleman_liau = r.coleman_liau().score   # .score or .grade_level
            metrics_dict[fn] = coleman_liau
        except:
            metrics_dict[fn] = None

        fn = prefix + 'ari'
        try:
            ari = r.ari().score   # .score or .grade_level
            metrics_dict[fn] = ari
        except:
            metrics_dict[fn] = None

        # requires 30 sentences so may give an error
        fn = prefix + 'smog'
        try:
            smog = r.smog().score   # .score or .grade_level
            metrics_dict[fn] = smog
        except:
            metrics_dict[fn] = None

    # FInal
    print(metrics_dict)

    # Return all the metrics

    return metrics_dict


def remove_nones(data):
    return [i for i in data if i is not None]


def dist_list_stats(data, title, x_label, filename):
    data = remove_nones(data)

    print('\n\n')
    print(title)
    print('Length after nones removed')
    print(len(data))
    print('MIN')
    print(min(data))
    print('MEAN')
    print(statistics.mean(data))
    print('MAX')
    print(max(data))
    graphs.histogram(data, title, x_label, 'Frequency', filename, bin_num=20, log_y=False)


# MAIN
def main(args):

    if ANALYSE_BERT:

        print('** Loading results csv')
        file = f'{args.input_dir}/' + FILE_NAME
        results_df = pandas.read_csv(file)
        file = f'{args.input_dir}/sents_' + FILE_NAME
        segmented_df = pandas.read_csv(file)

        hallucinated_entity_types = []
        results = []

        for idx, case in results_df.iterrows():
            print('*** Analysing case ' + str(idx))

            gold = case['gold']
            source = case['source']
            generated = case['pegasus_output']

            gold_segmented = segmented_df.iloc[idx]['gold_sents'].split('[SENTMARKER]')
            generated_segmented = segmented_df.iloc[idx]['pegasus_output_sents'].split('[SENTMARKER]')

            # If chain or not
            metrics_dict = summary_eval(generated, generated_segmented, gold, gold_segmented, source, CHAIN, RUN_TYPE)
            results.append(metrics_dict)

        #### Aggregrate analysis
        results_df = pandas.DataFrame(results)
        fn = f'{args.output_dir}/' + RUN_NAME + '_metrics.csv'
        results_df.to_csv(fn)

        # iterate over each column
        for col_name, col_data in results_df.items():
            print('** Analysing column: ' + str(col_name))
            data = col_data.values
            fn = f"{args.output_dir}/" + RUN_NAME + '_dist_' + col_name
            dist_list_stats(data, str(col_name), str(col_name), fn)

        # Pie chart of hallucinated entity types
        # THIS IS LIKELY TO HAVE ERRORS AS USING EXACT MATCH SO DON'T REPORT DIRECTLY,
        # JUST TO INFORM INVESTIGATION
        hallucinated_entity_types_dict = dict(Counter(hallucinated_entity_types))
        print(hallucinated_entity_types_dict)

    if ANALYSE_OREO:    # do it all again with oreo version
        print('**** Analysing with oreo version...')
        print('** Loading results csv')
        file = f'{args.input_dir}/' + FILE_NAME
        results_df = pandas.read_csv(file)
        file = f'{args.input_dir}/sents_' + FILE_NAME
        segmented_df = pandas.read_csv(file)

        hallucinated_entity_types = []
        results = []

        for idx, case in results_df.iterrows():
            print('*** Analysing case ' + str(idx))

            gold = case['gold']
            source = case['source']
            generated = case['oreo_pegasus_output']

            gold_segmented = segmented_df.iloc[idx]['gold_sents'].split('[SENTMARKER]')
            generated_segmented = segmented_df.iloc[idx]['oreo_pegasus_output_sents'].split('[SENTMARKER]')

            # If chain or not
            metrics_dict = summary_eval(generated, generated_segmented, gold, gold_segmented, source, CHAIN, RUN_TYPE)
            results.append(metrics_dict)

        #### Aggregrate analysis
        results_df = pandas.DataFrame(results)
        fn = f'{args.output_dir}/' + RUN_NAME + '_OREO_metrics.csv'
        results_df.to_csv(fn)

        # iterate over each column
        for col_name, col_data in results_df.items():
            print('** Analysing column: ' + str(col_name))
            data = col_data.values
            fn = f"{args.output_dir}/" + RUN_NAME + '_OREO_dist_' + col_name
            dist_list_stats(data, str(col_name), str(col_name), fn)

        # Pie chart of hallucinated entity types
        # THIS IS LIKELY TO HAVE ERRORS AS USING EXACT MATCH SO DON'T REPORT DIRECTLY,
        # JUST TO INFORM INVESTIGATION
        hallucinated_entity_types_dict = dict(Counter(hallucinated_entity_types))
        print(hallucinated_entity_types_dict)

    # cleanup
    tool.close()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
