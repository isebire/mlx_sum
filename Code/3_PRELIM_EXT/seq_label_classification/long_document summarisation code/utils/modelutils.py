import torch
import os
import csv
import numpy as np
from scipy.special import softmax
import math
from fairseq.models.roberta import RobertaModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertModel
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nlgeval import NLGEval  #https://github.com/Maluuba/nlg-eval
import re

def initializeModel(model_type, gpu=False):
    if model_type == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
        model = GPT2LMHeadModel.from_pretrained('models/gpt2-large')
    elif model_type == "roberta":
        model = RobertaModel.from_pretrained('models/roberta/roberta.large.mnli',
                                             checkpoint_file='model.pt')
        tokenizer = None
    elif model_type == "baselines-bert":
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        model = BertModel.from_pretrained('bert-large-uncased')
    else:
        return None, None
    if gpu:
        model = model.to('cuda')
    model.eval()
    return model, tokenizer

def import_bert_embeddings(allsentences, model, tokenizer, gpu=False):
    print("Building BERT embeddings for total sentences:", len(allsentences))
    emb = np.asarray([])
    for iter, sent in enumerate(allsentences):
        bert_tokens_sentence = tokenizer.encode(sent, add_special_tokens=True)
        bert_embeddings = np.asarray([])
        for i in range(0, len(bert_tokens_sentence), 512):
            with torch.no_grad():
                tensor_input = torch.tensor([bert_tokens_sentence[i:i + 512]])
                if gpu:
                    tensor_input = tensor_input.to('cuda')
                embeddings = model(tensor_input)[0].squeeze(0).cpu().numpy()
                if bert_embeddings.shape[0] == 0:
                    bert_embeddings = embeddings
                else:
                    bert_embeddings = np.append(bert_embeddings, embeddings, axis=0)

        f_emb_avg = np.mean(bert_embeddings, axis=0)
        if len(emb) == 0:
            emb = np.array([f_emb_avg])

        else:
            emb = np.append(emb, [f_emb_avg], axis=0)
    return emb

def getGpt2Perplexity(model, tokenizer, context, text, gpu=False):
    # PADDING_TEXT = context + "<|endoftext|>"
    PADDING_TEXT = context
    tokenize_input = tokenizer.tokenize(PADDING_TEXT + text)
    tokenize_text = tokenizer.tokenize(text)

    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    if gpu:
        tensor_input = tensor_input.to('cuda')
    with torch.no_grad():
        outputs = model(tensor_input, labels=tensor_input)
        loss, logits = outputs[:2]
    lp = 0
    perplexity_arr = ""
    for i in range((len(tokenize_input) - len(tokenize_text)), (len(tokenize_input) - 1)):
        masked_index = i
        predicted_score = logits[0, masked_index]
        predicted_prob = softmax(np.array(predicted_score.cpu()))
        probs = predicted_prob[tokenizer.convert_tokens_to_ids([tokenize_input[i + 1]])[0]]
        perplexity_arr += tokenize_input[i] + " " + str(logits) + ", "
        lp += np.log(probs)

    perplexity = math.exp(-lp / len(tokenize_text))
    return perplexity, perplexity_arr

def getRobertaEntailment(model, tokenizer, context, text, gpu=False):
    # gives prob that 2nd arg is entailed, contradicted, etc. by 1st arg.
    # argmax() == 2 means entailment

    tokens = model.encode(context, text)

    if len(tokens) > 512:
        tokens = tokens[:512]

    probs = model.predict('mnli', tokens)
    probs = probs.cpu().detach().numpy()[0]
    label = model.predict('mnli', tokens).argmax()
    label = label.cpu().item()
    prob_label = probs[label]
    return str(label), str(prob_label)

def getCosineScore(a, b):
    return np.dot(a, b) / (np.sqrt(np.dot(a, a)) * np.sqrt(np.dot(b, b)))

def getMultipleInputs(article, numInputs):
    # Combine every 'numInputs' article sentences, leave the rest as it is.
    new_article = []
    temp = ""
    for i in range(0, len(article)):
        if i % numInputs == 0:
            if len(temp) > 0:
                new_article.append(temp)
            temp = ""
        temp = temp + str(article[i]) + " "

    if len(temp) > 0:
        new_article.append(temp)

    return new_article

def preprocessSentence(str_):
    regex = re.compile('[*▪_\"\"•]') 
    str_ = re.sub(regex,'', str_)
    str_ = str_.replace('\n', '').replace('\r', '')
    return str_

def isinvalidElseModify(a, s):
    if s is None or a is None:
        return True, a, s
    s = preprocessSentence(s)
    a = preprocessSentence(a)
    if (len(s.strip()) < 5) or (len(a.strip()) < 5):
        return True, a, s
    elif len(a.split()) < 10 or len(s.split())<5:
    # less than 5 words
        #print(a,"||",s)
        return True, a, s
    else:
        return False, a, s

def processPairGPT2(pair, model, tokenizer, gpu, output_dir, index, numInputs):
    # fout = open(os.path.join(output_dir, 'sample' + '_' + str(index) + "_" + str(numInputs) + '.csv'), 'w')
    fscores = open(os.path.join(output_dir, 'scores' + '_' + str(index) + "_" + str(numInputs) + '.csv'), 'w')
    article, summary = pair
    article = article.split("\n")
    summary = summary.split("\n")
    print('Article length: ', len(article))
    print('Summary length: ', len(summary))
    new_article = getMultipleInputs(article, numInputs)
    print('Final Article length (after numInputs): ', len(new_article))
    wcscores = csv.writer(fscores)
    wcscores.writerow(['id', 'context_id', 'context', 'input', 'preplexity'])
    count = 0
    for sindex, s in enumerate(summary):
        for aindex, a in enumerate(new_article):
            flag,a,s = isinvalidElseModify(a,s)
            if flag:
                continue
            perplexity, perplexity_arr = getGpt2Perplexity(model, tokenizer, a, s, gpu)
            wcscores.writerow([sindex, aindex,  a, s, perplexity])
            count += 1
        print('Writing for summary sentence', sindex)
    print('Total pairs:', count)
    return

def processPairRoberta(pair, model, tokenizer, gpu, output_dir, index, numInputs):
    fscores = open(os.path.join(output_dir, 'scores' + '_' + str(index) + "_" + str(numInputs) + '.csv'), 'w')
    article, summary = pair
    article = article.split("\n")
    summary = summary.split("\n")
    print('Article length: ', len(article))
    print('Summary length: ', len(summary))
    new_article = getMultipleInputs(article, numInputs)
    print('Final Article length (after numInputs): ', len(new_article))
    wcscores = csv.writer(fscores)
    wcscores.writerow(['id', 'context_id', 'context', 'input', 'type', 'prob'])
    count = 0
    for sindex, s in enumerate(summary):
        for aindex, a in enumerate(new_article):
            flag,a,s = isinvalidElseModify(a,s)
            if flag:
                continue
            type, prob = getRobertaEntailment(model, tokenizer, a, s, gpu)
            wcscores.writerow([sindex, aindex, a, s, type, prob])
            count += 1
        print('Writing for summary sentence', sindex)
    print('Total pairs:', count)
    return

def processPairBERTEmb(pair, model, tokenizer, gpu, output_dir, index, numInputs):
    fscores = open(os.path.join(output_dir, 'scores' + '_' + str(index) + "_" + str(numInputs) + '.csv'), 'w')
    article, summary = pair
    article = article.split("\n")
    summary = summary.split("\n")
    print('Article length: ', len(article))
    print('Summary length: ', len(summary))
    new_article = getMultipleInputs(article, numInputs)
    print('Final Article length (after numInputs): ', len(new_article))
    wcscores = csv.writer(fscores)
    wcscores.writerow(['id', 'context_id', 'context', 'input', 'similarity'])
    source_emb = import_bert_embeddings(new_article, model, tokenizer, gpu)
    print('Got source emb for doc pair: ', index)
    target_emb = import_bert_embeddings(summary, model, tokenizer, gpu)
    print('Got target emb for doc pair: ', index)
    print('Got all embeddings for doc pair: ', index)
    print(source_emb.shape, target_emb.shape)
    count = 0
    for i, s in enumerate(summary):
        for j, a in enumerate(new_article):
            flag,a,s = isinvalidElseModify(a,s)
            if flag:
                continue
            assert (target_emb[i, :].shape, source_emb[j, :].shape)
            score = getCosineScore(target_emb[i, :], source_emb[j, :])
            wcscores.writerow([i, j, a, s, score])
            count+=1
        print('Writing for summary sentence', i)
    print('Total pairs:', count)
    return

def processPairNGRAM(pair, model, tokenizer, gpu, output_dir, index, numInputs):
    fscores = open(os.path.join(output_dir, 'scores' + '_' + str(index) + "_" + str(numInputs) + '.csv'), 'w')
    article, summary = pair
    article = article.split("\n")
    summary = summary.split("\n")
    print('Article length: ', len(article))
    print('Summary length: ', len(summary))
    new_article = getMultipleInputs(article, numInputs)
    print('Final Article length (after numInputs): ', len(new_article))
    wcscores = csv.writer(fscores)
    wcscores.writerow(['id', 'context_id', 'context', 'input', 'Bleu_1', 'Bleu_2','Bleu_3','Bleu_4','METEOR','ROUGE_L','CIDEr',
                       'SkipThoughtCS','EmbeddingAverageCosineSimilarity','VectorExtremaCosineSimilarity','GreedyMatchingScore'])
    nlgeval = NLGEval()
    count = 0
    for sindex, s in enumerate(summary):
        for aindex, a in enumerate(new_article):
            flag,a,s = isinvalidElseModify(a,s)
            if flag:
                continue
            metrics= nlgeval.compute_individual_metrics([a], s)
            wcscores.writerow([sindex, aindex, a, s, metrics['Bleu_1'],metrics['Bleu_2'],metrics['Bleu_3'],metrics['Bleu_4'], metrics['METEOR'], 
                              metrics['ROUGE_L'], metrics['CIDEr'], metrics['SkipThoughtCS'], metrics['EmbeddingAverageCosineSimilarity'],
                              metrics['VectorExtremaCosineSimilarity'], metrics['GreedyMatchingScore']])
            count+=1
        print('Writing for summary sentence', sindex)
    print('Total pairs:', count)
    return

def processPairBLEU(pair, model, tokenizer, gpu, output_dir, index, numInputs):
    fscores = open(os.path.join(output_dir, 'scores' + '_' + str(index) + "_" + str(numInputs) + '.csv'), 'w')
    article, summary = pair
    article = article.split("\n")
    summary = summary.split("\n")
    print('Article length: ', len(article))
    print('Summary length: ', len(summary))
    new_article = getMultipleInputs(article, numInputs)
    print('Final Article length (after numInputs): ', len(new_article))
    wcscores = csv.writer(fscores)
    wcscores.writerow(['id', 'context_id','context', 'input', 'Blue score'])
    bleu = sentence_bleu
    cc = SmoothingFunction()
    count = 0
    #breakpoint()
    for sindex, s in enumerate(summary):
        for aindex, a in enumerate(new_article):
            flag,a,s = isinvalidElseModify(a,s)
            if flag:
                continue
            ref = a.split()
            hyp = s.split()
            score = bleu([ref], hyp, smoothing_function=cc.method3)
            wcscores.writerow([sindex, aindex, a, s, score])
            count+=1
        print('Writing for summary sentence', sindex)
    print('Total pairs:', count)
    return


