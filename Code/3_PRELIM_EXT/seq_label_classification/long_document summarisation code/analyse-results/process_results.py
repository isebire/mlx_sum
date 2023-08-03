#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
import re



def getSummarySentence(df_dict, id_dict, id, outFile):
    for model in df_dict.keys():
        df = df_dict[model]
        [min_id_model, max_id_model] = id_dict[model]
        if id >= min_id_model and id<= max_id_model:
            df = df[df['id'] == id]
            if df['input'].shape[0] == 0:
                continue # that summary id not found in particular model file
            else:
                return df.iloc[0]['input']

def isinvalid(sent):
    
    if sent is None:
        return True
    
    # less than 10 words
    if len(sent.split()) < 10:
        return True
    
    regex = re.compile('[*▪]') 
          
    if(regex.search(sent) != None): 
        return True
        
    return False


# In[69]:


models = ['roberta-entailment', 'gpt2-perplexity', 'baseline-bert', 'baseline-bleu'] # Models to compare against

dataset = 'beige_books'  # beige_books, minutes, amicus_facts, amicus
nInputs = 1  # Configuration of how many i/p sentences to consider
topk = 3 # max num of input sentences (per summary sentence)

files = os.listdir("inputs/" + models[0] + "/" + dataset)  # results list
docs = []

for file in files:
     if file.endswith(".csv"):
        docs.append(file)
        
print(docs)
print('Running for {0} documents in {1} dataset, with nInputs = {2}'.format(len(docs), dataset, str(nInputs)))
outdir = "results/" + dataset + "/" + "numInputs_" + str(nInputs)

if not os.path.exists(outdir):
    os.makedirs(outdir)


for doc in docs:
    print('---------')
    print('Document ID: ', doc[8:len(doc)-6])
    print('---------')
    
    df_dict = {}
    id_dict = {}
    minid = 1e7
    maxid = -1e7
    summary_sentences = {}
    
    filename = doc
    for model in models:
        file = "inputs/" + model + "/" + dataset + "/" + filename
        df = pd.read_csv(file)
        idList = df['id'].to_list()
        if len(idList) != 0:
            minid = min(minid, min(idList))
            maxid = max(maxid, max(idList))
        df_dict[model] = df
        id_dict[model] = [minid, maxid]
        
    outFile = open(outdir + "/" + str(doc[8:len(doc)-6]) + '.txt', 'w')
    outCsv = outdir + "/" + str(doc[8:len(doc)-6]) + '.csv'
    
    sid = []
    summSents = []
    inputs = {}
    
    for model in models:
        # init empty list
        inputs[model] = []
        
        
    for id in range(minid, maxid + 1):
        summarySent = getSummarySentence(df_dict, id_dict, id, outFile)
        if isinvalid(summarySent):
            continue
#         print('---------')
#         print('Summary Sentence ID: ', id)
#         print(summarySent, "\n")
#         print('---------')
        print('---------', file = outFile)
        print('Summary Sentence ID: ', id, file = outFile)
        print(summarySent, "\n", file = outFile)
        print('---------', file = outFile)
    
        for k in range(topk):
            sid.append(id)
            summSents.append(summarySent)
            
        for model in models:
#             print(model)
#             print('---------')
            print(model, file = outFile)
            print('---------', file = outFile)
            
            [min_id_model, max_id_model] = id_dict[model]
            
            if id >= min_id_model and id<= max_id_model:
                df = df_dict[model]
                df = df[df['id'] == id]
                
                k_results = df.shape[0]
                for r in range(k_results):
                    sent = df.iloc[r]['context']
                    
#                     print(sent)
#                     print("\n")
                    print(sent, file = outFile)
                    print("\n", file = outFile)
                    
                    inputs[model].append(sent) # need to have exactly topk entries in each list (csv format)
                
                for k in range(k_results, topk):
                    inputs[model].append("-")
            
            else:
                # model had no valid input sentences
                for k in range(topk):
                    inputs[model].append("-")
                    
#                 print('---------')
                print('---------', file = outFile)
    
    # listsize of all models should be same
    values = []
    for val in inputs.values():
        # each val is a list (for one model's outputs)
        values.append(len(val))
    assert(len(set(values)) == 1)
    
    outFile.close()
    
    # Dump CSV
    data = []
    headers = ['sid', 'summary_sent']
    data.append(pd.Series(sid))
    data.append(pd.Series(summSents))
    
    for key in inputs.keys():
        headers.append(key) # model name
        data.append(pd.Series(inputs[key]))
    
    headers.append('Score')
    data.append(pd.Series())
    
    df = pd.concat(data, axis=1, keys=headers)
    df.to_csv(outCsv, index = False)
        
print("Done Processing..")
    


# In[ ]:




