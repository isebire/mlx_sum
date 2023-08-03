# Abstractive-Summarization

Abstractive summarization is the task of compressing a long document into a coherent short document while retaining salient information. Modern abstractive summarization methods are based on deep neural networks which are data hungry algorithms. However, collecting summarization datasets is an expensive and time-consuming task and practical industrial settings are usually \emph{low-resource}. In this paper, we study a challenging low-resource setting of summarizing long legal briefs with average source document length 4268 words and only 120 (document, summary) pairs. A state-of-the-art abstractive summarization BART~\citep{lewis2019bart} system is only able to achieve 17.9 ROUGE-L, noticing that it struggles with long documents. We attempt to compress these long documents by identifying salient sentences in the source which best ground the summary, using a novel algorithm based on GPT2 perplexity scores. These sentences are used to train a sentence salience classifier, which compresses the source document significantly. When these compressed documents are fed to a pre-trained BART model, we observe a 6 ROUGE-L F1 points over several baselines. Furthermore, qualitative inspection shows that the classifier tends to recover sentences that humans find salient for the summary on a subset of the data labeled by domain experts.

# Folder structure

analyse-results: contains scripts to take a scores file and transform it by picking top source sentences for each summary sentence for analysis.  
 
baseline-bleu: code for BLEU computation between all source, summary sentence pairs.  
 
baseline-ngram: code for ngram metrics such as ROUGE, METEOR, BLEU etc for all source, summary sentence pairs  
 
bert-classifier: code for BERT classifier used as Sentence Salience Classifier.  
 
baselines-bert: code for cosine similarity computation using BERT CLS embedings or all source, summary sentence pairs   
 
data-splitting: Given scores computed using above methods, to sample positive/negative examples using sampling method (aggregate method, random method, topk-bottomk method) and split into train, test, dev split at a document level.
 
eval: ROUGE evaluation code. 

generate-sentence-dumps: Using spacy to generate sentence splits for each file. 
 
gpt2-perplexity: code for perplexity computation using  GPT2 or all source, summary sentence pairs.
 
infer-bart: bart inference and finetuning sripts
 
preprocess-amicus: code to automatically extract summary and target
 
roberta-entailment: code for entailment computation using RoBERTa for all source, summary sentence pairs 

text-rank: Text-rank baseline code
 
utils: All utility functions used for different puposes.

# Pipeline

Assuming the data is present in a data folder with all pairs in different files as 0.src, 0.tgt, 1.src, 1.tgt etc
 
1. Sentence splitting:   

```
python generate-sentence-dumps/generate.py --sourceDir=${SOURCE_DIR} --outputDir=${SENT_DIR} --datatype='amicus'
```

2. Running diff models(BERT, BLEU, ROBERTA, GPT2)
```
python $model_dir$/map-summary-reference.py --sourceDir=${SENT_DIR} --outputDir=${SCORES_DIR} --gpu
```

3. Preparing training data for classifier using sampling method:  
```
python data-splitting/generate.py --sourceDir=${SCORES_DIR} --outputDir=${CLASSIFIER_DIR} --dataType='amicus' --field='similarity' --criterion='max' --method='topk' --createSplit --readSplit
```

4. Training Classifier and inference: Follow README.md in bert-classifier folder
 
5. Processing classifier output on test set to BART input style: call generateTestSrcFilesForInferring() in utils/miscutils.py with necessary params.

6. Infer BART

7. ROUGE calculation on target, generated_summary( or hypo)
