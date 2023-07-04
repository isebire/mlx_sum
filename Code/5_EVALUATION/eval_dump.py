# Evaluation dump - assuming gold and pegasus summaries

from rouge_score import rouge_scorer
from datasets import load_dataset, load_from_disk
import GRAPHS as graphs
import statistics
from bert_score import score
from faithfulness.Entailment import Entailment, EntailmentMethod
from readability import Readability
import language_tool_python]
import pandas

# Still need to finish implementing metrics and data structures, and test

# Key
# !! To do
# $$ To test

def summary_eval(generated_summary, gold_summary, source):

    metrics_dict = {}

    # Record of all columns - starred is reported in OG
    # r1_precision, r1_recall, *r1_f1
    # r2_precision, r2_recall, *r2_f1
    # rL_precision, rL_recall, *rL_f1
    # bs_precision, bs_recall, *bs_f1
    # grammatical_errors
    # $$ entailment_precision, entailment_recall, entailment_f1   (avg across all sents in summary)
    # flesch_kincaid, coleman_liau, ari, smog

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

    # BERTScore
    bs_precision, bs_recall, bs_f1 = score([generated_summary], [gold_summary], lang="en", verbose=True)
    metrics_dict['bs_precision'] = bs_precision
    metrics_dict['bs_recall'] = bs_recall
    metrics_dict['bs_f1'] = bs_f1

    # !! Unique ngram ratio

    # !! Normalised inverse of diversity

    # !! More than sentiments redundancy

    # Grammaticality (language tool)
    errors = tool.check(generated_summary)
    # It can also correct = tool.correct(text)
    metrics_dict['grammatical_errors'] = len(errors)

    ## CHAIN ANALYSIS (where applicable)

    # !! ROUGE between reference and pegasus entity chains in applicable cases

    # !! Entity precision /acc/F1 between pegasus chain and reference chain???
    # !! Entity precision / acc / F1 between pegasus chain and source?
    # !! Entity precision  / acc / F1 between pegasus summary and pegasus chain

    ## $$ ENTAILMENT (faithfulness library)

    # use bart model with PEGASUS INPUT (not sources) (it is already less than 1024 tokens)
    # so do not need chunking as use 'source' as what is pegasus input

    # !! If using faithfulness library - pip install faithfulness. NEEDS TO RUN ON CLUSTER AS CUDA

    # in thesis paper - calculating wrt document better than wrt sentence
    entailment_metric = Entailment(method=EntailmentMethod.DOC)

    # segement summary into sentences. lexnlp neeeded??
    generated_summary_sentences = []
    entail_precs = []
    entail_recalls = []
    entail_f1s = []

    for generated_summary_sentence in generated_summary_sentences:
        entailment_result = entailment_metric.score(generated_summary_sentence, source)
        entail_precs.append(entailment_result['precision'])
        entail_recalls.append(entailment_result['recall'])
        entail_f1s.append(entailment_result['f1'])

    # don't really get why there are all 3???? test and look into format of
    # direct result from entailment
    # https://github.com/bigabig/faithfulness/blob/main/src/faithfulness/Entailment.py

    # Mean across all sents in the summary
    metrics_dict['entailment_precision'] = statistics.mean(entail_precs)
    metrics_dict['entailment_recall'] = statistics.mean(entail_recalls)
    metrics_dict['entailment_f1'] = statistics.mean(entail_f2s)

    # !! also do for gold summaries!!

    ## READABILITY
    # Do for: original sources, gold summary, pegasus summaries
    r = Readability(text)   # must contain at least 100 words
    metrics_dict['flesch_kincaid'] = r.flesch_kincaid().score   # .score or .grade_level
    metrics_dict['coleman_liau'] = r.coleman_liau().score  #.score, .grade_level
    metrics_dict['ari'] = r.ari().score #.score, .grade_levels, .ages
    metrics_dict['smog'] - r.smog(all_sentences=True).score  #.score, .grade_level

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

print('*** Loading results csv')  # *** NOTE FILENAME AND COLUMN NAMES WILL DIFFER
results_df = pandas.read_csv('outputs_test.csv')


print('** Loading eval utils...')
scorer_rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
tool = language_tool_python.LanguageTool('en-US')


#### Actual eval metrics - do for each case and save to some data structure

results = {}

for idx, case in results_df.iterrows():

    # Split into summary and entity chain if needed!
    generated_summary = case['pegasus_output']
    gold_summary = case['gold']
    source = case['source']

    metrics_dict = summary_eval(generated_summary, gold_summary, source)
    results.append(metrics_dict)

#### Aggregrate analysis
results_df = pandas.DataFrame(results)




# dist_list_stats for aggregate list

# Correlation of summary quality eg rougeLF1 with number of source documents / source document length?

# could do box plot of dists with different models
# graphs.box_plot(data, labels, title, x_label, y_label, filename)



tool.close()
