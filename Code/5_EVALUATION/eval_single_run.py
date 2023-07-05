# Evaluation dump - assuming gold and pegasus summaries
# This is for a single one of the pegasus runs
# SET RUN NAME BELOW

from rouge_score import rouge_scorer
from datasets import load_dataset, load_from_disk
import GRAPHS as graphs
import statistics
from bert_score import score
from faithfulness.Entailment import Entailment, EntailmentMethod   # NOTE: using custom edited ver, not pkg!
from readability import Readability
import language_tool_python
import pandas
import MoreThanSentiments as mts  # pip install MoreThanSentiments

# Still need to finish implementing metrics and data structures, and TEST!!!

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
    frequencies = ((i / len(words)) for i in counts.values())
    entropy = - sum(f * log(f, 2) for f in frequencies)

    nid = 1 - (entropy / log(len(words))
    return nid

def summary_eval(generated_summary, gold_summary, source):

    metrics_dict = {}

    # Record of all columns - starred is reported in OG
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

    # BERTScore (standard - for comparison with og???)
    bs_precision, bs_recall, bs_f1 = score([generated_summary], [gold_summary], lang="en", verbose=True)
    metrics_dict['bs_precision'] = bs_precision
    metrics_dict['bs_recall'] = bs_recall
    metrics_dict['bs_f1'] = bs_f1

    # BERTScore better human correlation and handling of longer lengths
    #  facebook/bart-large-mnli
    bs_precision, bs_recall, bs_f1 = score([generated_summary], [gold_summary], model_type='facebook/bart-large-mnli' lang="en", verbose=True)
    metrics_dict['bs_mnli_precision'] = bs_precision
    metrics_dict['bs_mnli_recall'] = bs_recall
    metrics_dict['bs_mnli_f1'] = bs_f1

    # !! Unique ngram ratio  - trigrams? n = 1,2,3 in paper with defs
    # count (unique n grams ) / count (n grams)
    all_trigrams = [generated_summary[i:i+3] for i in range(len(generated_summary)-2)]
    num_total_trigrams = len(all_trigrams)
    num_unique_trigrams = len(list(set(all_trigrams))
    metrics_dict['unique_trigram_ratio'] = num_unique_trigrams / num_total_trigrams

    # Normalised inverse of diversity
    # NID = 1 -   entropy(document) / log(len(document))
    metrics_dict['nid'] = nid(generated_summary)

    # More than sentiments redundancy
    # original: df['cleaned_data'] = df.text.apply(mts.clean_data, args=(True, True, False, True, False))
    mts_cleaned_text = mts.clean_data(generated_summary, True, True, False, True, False)
    redundancy = mts.Redundancy(mts_cleaned_text, n = 10)
    metrics_dict['redundancy']

    # Grammaticality (language tool)
    errors = tool.check(generated_summary)
    # It can also correct = tool.correct(text)
    metrics_dict['grammatical_errors'] = len(errors)

    ## CHAIN ANALYSIS (where applicable)

    # !! ROUGE between reference and pegasus entity chains in applicable cases

    # !! Entity precision /acc/F1 between pegasus chain and reference chain???
    # !! Entity precision / acc / F1 between pegasus chain and source?
    # -> + by entity category?

    # !! Entity precision  / acc / F1 between pegasus summary and pegasus chain

    ## ENTAILMENT (faithfulness library)

    # facebook/bart-large-mnli  (same as new BERTScore)
    # it is already less than 1024 tokens as 'source' as what is pegasus input
    # so do not need chunking

    entailment_metric = Entailment(method=EntailmentMethod.DOC)

    # segement summary into sentences. lexnlp neeeded??
    generated_summary_sentences = []
    entailments = []

    for generated_summary_sentence in generated_summary_sentences:
        entailment_result = entailment_metric.score(generated_summary_sentence, source)
        entailments.append(entailment_result)

    # Mean across all sents in the summary
    metrics_dict['pegasus_entailment'] = statistics.mean(entailments)

    # also do for gold summaries!!

    gold_summary_sentences = []
    entailments = []

    for gold_summary_sentence in gold_summary_sentences:
        entailment_result = entailment_metric.score(gold_summary_sentence, source)
        entailments.append(entailment_result)

    # Mean across all sents in the summary
    metrics_dict['gold_entailment'] = statistics.mean(entailments)


    ## READABILITY
    # Do for: original sources, gold summary, pegasus summaries
    for text in [generated_summary, gold_summary]:
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


    # Return all the metrics

    return metrics_dict

def dist_list_stats(data, title, x_label, filename):
    print('\n\n')
    print(title)
    print('MIN')
    print(min(data))
    print('MEAN')
    print(statistics.mean(data))
    print('MAX')
    print(max(data))
    graphs.histogram(data, title, x_label, 'Frequency', filename, bin_num=20, log_y=False)


# MAIN
RUN_NAME

print('*** Loading results csv')  # *** NOTE FILENAME AND COLUMN NAMES WILL DIFFER
results_df = pandas.read_csv('outputs_test.csv')


print('** Loading eval utils...')
scorer_rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
tool = language_tool_python.LanguageTool('en-US')


#### Actual eval metrics - do for each case and save to some data structure

results = []

for idx, case in results_df.iterrows():

    # Split into summary and entity chain if needed! !!!
    generated_summary = case['pegasus_output']
    gold_summary = case['gold']
    source = case['source']

    metrics_dict = summary_eval(generated_summary, gold_summary, source)
    results.append(metrics_dict)

#### Aggregrate analysis
results_df = pandas.DataFrame(results)
fn = RUN_NAME + '_metrics.csv'
results_df.to_csv(fn)

# iterate over each column
for (col_name, col_data) in results_df.iteritems():
    print('** Analysing column: ' + str(col_name))
    data = col_data.values
    fn = 'dist_' + col_name
    dist_list_stats(data, title, x_label, fn)

tool.close()
