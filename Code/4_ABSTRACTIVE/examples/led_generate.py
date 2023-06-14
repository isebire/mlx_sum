# LED generate from https://github.com/Law-AI/summarization/tree/aacl/abstractive/Legal-LED

dataset = "IN" # Options: IN - IN-Abs, UK-UK-Abs, N2-IN-Ext
output_path = "./output/"

import pandas as pd
import numpy as np
import glob
import sys
sys.path.insert(0, '../')
from utilities import *
import os
import nltk
import os
import re
import numpy as np
import pandas as pd
import json
import random
import nltk
nltk.download('punkt')

from IPython.display import display, HTML
import torch
import datasets
from datasets import load_dataset, load_metric, Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import LEDTokenizer, LEDForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

if not os.path.exists(output_path):
    os.makedirs(output_path)

#Reading the test documents
names, data_source, data_summary = get_summary_data(dataset, "test")
print(len(names))
print(len(data_source))
print(len(data_summary))
len_dic = dict_names = get_req_len_dict(dataset, "test")

model_name = "nsi319/legal-led-base-16384"
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoder_max_length = 1024*16
decoder_max_length = 1024
led = AutoModelForSeq2SeqLM.from_pretrained(model_name, gradient_checkpointing=True, use_cache=False)
# set generate hyperparameters
led.config.num_beams = 2
led.config.max_length = decoder_max_length
led.config.min_length = 256
led.config.early_stopping = True
led.config.no_repeat_ngram_size = 4
led = AutoModelForSeq2SeqLM.from_pretrained("./final_model/IN_model", use_cache=False)
led = led.to("cuda")


import torch
led.eval()

def generate_summary_gpu(input_text, l):
	inputs = tokenizer(
		input_text,
		padding="max_length",
		truncation=True,
		max_length=encoder_max_length,
		return_tensors="pt",
	)
	input_ids = inputs.input_ids.to(led.device)
	attention_mask = inputs.attention_mask.to(led.device)
	global_attention_mask = torch.zeros_like(attention_mask)
	# put global attention on <s> token
	global_attention_mask[:, 0] = 1
	l = max(l,256)
	l = min(l,1024)
	outputs = led.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask, max_length = l, num_beams = 2,repetition_penalty = 2.5,early_stopping = True)
	preds = [tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_space=True) for gen_id in outputs]
	return "".join(preds)

for i in range(len(data_source)):
    name = names[i]
    doc = data_source[i]
    wc = doc.split(" ")
    input_len = len(wc)
    req_len = dict_names[name]
    print(str(i) + ": " + name +  " - " + str(input_len) + " : " + str(req_len))

    abs_summ = generate_summary_gpu(doc,req_len)
    if len(abs_summ.split(" ")) > req_len:
        abs_summ = abs_summ.split(" ")
        abs_summ = abs_summ[:req_len]
        abs_summ = " ".join(abs_summ)
    print(abs_summ)
    print(len((abs_summ.split(" "))))
    path = output_path + name
    file = open(path,'w')
    file.write(abs_summ)
    file.close()
