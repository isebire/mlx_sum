# Legal pegasus generate summaries from https://github.com/Law-AI/summarization/tree/aacl/abstractive/Legal-Pegasus

import pandas as pd
import numpy as np
import glob
import sys
sys.path.insert(0, '../')
from utilities import *
import os
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Trainer, TrainingArguments

dataset = "IN" # Options: IN - IN-Abs, UK-UK-Abs, N2-IN-Ext
output_path = "./output/"

def summerize(text, max_len, min_len):
    '''
    Function to generate summary using Pegasus
    input:  nested_sentences - chunks
            max_l - Maximum length
            min_l - Minimum length
    output: document summary
    '''
    try:
        input_tokenized = tokenizer.encode(text, return_tensors='pt',max_length=512,truncation=True).to(device)
        summary_ids = model.generate(input_tokenized,
                                          num_beams=9,
                                          length_penalty=0.1,
                                          min_length=min_len,
                                          max_length=max_len,
                                    )
        summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]
        return summary
    except:
        return ""


def summerize_doc(nested_sentences, p):
    '''
    Function to generate summary using chunking based Pegasus
    input:  nested_sentences - chunks
            p - Number of words in summaries per word in the document
    output: document summary
    '''
    device = 'cuda'
    result = []
    for nested in nested_sentences:
        l = int(p * len(nested.split(" ")))
        max_len = l
        min_len = l-5
        result.append(summerize(nested, max_len, min_len))
    return result

if not os.path.exists(output_path):
    os.makedirs(output_path)

#Reading the test documents
names, data_source, data_summary = get_summary_data(dataset, "test")
print(len(names))
print(len(data_source))
print(len(data_summary))
len_dic = dict_names = get_req_len_dict(dataset, "test")
device = "cuda:1"
tokenizer = AutoTokenizer.from_pretrained("nsi319/legal-pegasus")
model = AutoModelForSeq2SeqLM.from_pretrained("nsi319/legal-pegasus").to(device)

done_files = glob.glob(output_path + "*.txt")
done_files = [i[i.rfind("/")+1:] for i in done_files]

# main loop to generate and save summaries of each document in the test dataset
for i in range(len(data_source)):
    done_files = glob.glob(output_path + "*.txt")
    done_files = [i[i.rfind("/")+1:] for i in done_files]
    name = names[i]
    if name in done_files:continue
    doc = data_source[i]
    input_len = len(doc.split(" "))
    req_len = dict_names[name]
    print(str(i) + ": " + name +  " - " + str(input_len) + " : " + str(req_len), end = ", ")

    nested = nest_sentences(doc,512)
    p = float(req_len/input_len)
    print(p)
    abs_summ = summerize_doc(nested,p)
    abs_summ = " ".join(abs_summ)
    print(len((abs_summ.split(" "))))

    if len(abs_summ.split(" ")) > req_len:
        abs_summ = abs_summ.split(" ")
        abs_summ = abs_summ[:req_len]
        abs_summ = " ".join(abs_summ)
    print(len((abs_summ.split(" "))))
    path = output_path + name
    file = open(path,'w')
    file.write(abs_summ)
    file.close()
#     break
