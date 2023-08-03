#path = datasets/multinli/multinli_0.9/multinli_0.9_train.jsonl 
from modelutils import initializeModel, getGpt2Perplexity
import jsonlines
import logging
import numpy as np
from tqdm import tqdm
import csv
#import matplotlib.pyplot as plt
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO)
logger = logging.getLogger(__name__)
gpu = True

def getExamples():
    path = "datasets/multinli/multinli_0.9/multinli_0.9_train.jsonl"
    f = jsonlines.open(path, "r")
    positive = []
    negative = []
    neutral = []
    while True:
        try:
            sample = f.read()
            if sample["gold_label"]=='entailment':
                positive+=[(sample["sentence1"], sample["sentence2"])]
            if sample["gold_label"]=='contradiction':
                negative+=[(sample["sentence1"], sample["sentence2"])]
            if sample["gold_label"]=='neutral':
                neutral+=[(sample["sentence1"], sample["sentence2"])]
        except EOFError:
            break
    logger.info(f'Got examples for all three classes!')
    return np.array(positive), np.array(negative), np.array(neutral)

"""
def plot_scores(scores_pos, scores_neg, scores_neu):
    fig = plt.figure()
    # multiple line plot
    df=pd.DataFrame({'x': np.arange(0,1000), 'entailment': scores_pos, 'contradiction': scores_neg, 'neutral': scores_neu })
    plt.plot( 'x', 'entailment', data=df, color='green', label='entailment')
    plt.plot( 'x', 'contradiction', data=df, color='red', label='contradiction')
    plt.plot( 'x', 'neutral', data=df, color='blue',label='neutral')
    plt.legend()
    plt.title('Perplexity distribution on an MNLI dataste')
    plt.ylabel('perplexity scores')
    plt.savefig('mnli_perplexity.png')
"""

if __name__=="__main__":
    model, tokenizer = initializeModel("gpt2", gpu=gpu)
    logger.info(f'Model and tokenizer intialised!')
    scores_pos, scores_neg, scores_neu = [], [], []
    scores_pos_full, scores_neg_full, scores_neu_full = [], [], []
    positive_full, negative_full, neutral_full = getExamples()
    f = open('sentence_log.csv', 'w')
    w = csv.writer(f)
    w.writerow(['iteration', 'sent1', 'sent2', 'tag', 'ppl'])
    for i in range(0,5):
        logger.info(f'Iteration {i}')
        idx = np.random.choice(np.arange(len(positive_full)), 1000)
        positive = positive_full[idx]
        idx = np.random.choice(np.arange(len(negative_full)), 1000)
        negative =negative_full[idx]
        idx = np.random.choice(np.arange(len(neutral_full)), 1000)
        neutral = neutral_full[idx]
        for p,n,nt in tqdm(zip(positive, negative, neutral)):
            scores_pos.append(getGpt2Perplexity(model, tokenizer, p[0], p[1], gpu=gpu)[0])
            w.writerow([i, p[0], p[1], 'entailment', scores_pos[-1]])
            scores_neg.append(getGpt2Perplexity(model, tokenizer, n[0], n[1], gpu=gpu)[0])
            w.writerow([i, n[0], n[1], 'contradiction', scores_neg[-1]])
            scores_neu.append(getGpt2Perplexity(model, tokenizer, nt[0], nt[1], gpu=gpu)[0])
            w.writerow([i, nt[0], nt[1], 'neutral', scores_neu[-1]])

        scores_pos_full.append([scores_pos])
        scores_neg_full.append([scores_neg])
        scores_neu_full.append([scores_neu])
        
    f.close()
    scores_pos_full = np.array(scores_pos_full)
    scores_neg_full = np.array(scores_neg_full)
    scores_neu_full = np.array(scores_neu_full)
    logger.info(f'Score computation done')
    np.save('pos_scores_5.npy', scores_pos_full)
    np.save('neg_scores_5.npy', scores_neg_full)
    np.save('neu_scores_5.npy', scores_neu_full)
    logger.info(f'Files saved')
    
    