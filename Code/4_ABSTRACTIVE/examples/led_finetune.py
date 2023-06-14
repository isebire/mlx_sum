# Legal LED finetune from https://github.com/Law-AI/summarization/tree/aacl/abstractive/Legal-LED

import os
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

os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
val_files = [] # Add the validation files to be used
rouge = load_metric("rouge")

def getData(tokenizer, dataPath, MAX_DOC_LEN, val = 0):
	documentPath = f'{dataPath}/judgement'
	summaryPath = f'{dataPath}/summary'
	dataset = {'document':[], 'summary':[]}
	count = 0
	for file in os.listdir(documentPath):
		count += 1
		if os.stat(f'{documentPath}/{file}').st_size == 0 or os.stat(f'{summaryPath}/{file}').st_size == 0:
			continue
		doc_in = open(f'{documentPath}/{file}', 'r', encoding='utf8')
		doc_lines = [line.strip() for line in doc_in.readlines()]
		summ_in = open(f'{summaryPath}/{file}', 'r', encoding='utf8')
		summ_lines = [line.strip() for line in summ_in.readlines()]
		if len(doc_lines) == 0 or len(summ_lines) == 0:
			continue

		# print(file, train_files[0], type(file))
		if val == 0 and file not in val_files:
			dataset['document'].append(' '.join(doc_lines))
			dataset['summary'].append(' '.join(summ_lines))
		if val == 1 and file in val_files:
			dataset['document'].append(' '.join(doc_lines))
			dataset['summary'].append(' '.join(summ_lines))

	df = pd.DataFrame(dataset)
	return df


def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(
        batch["document"],
        padding="max_length",
        truncation=True,
        max_length=encoder_max_length,
    )
    outputs = tokenizer(
        batch["summary"],
        padding="max_length",
        truncation=True,
        max_length=decoder_max_length,
    )

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    # create 0 global_attention_mask lists
    batch["global_attention_mask"] = len(batch["input_ids"]) * [
        [0 for _ in range(len(batch["input_ids"][0]))]
    ]

    # since above lists are references, the following line changes the 0 index for all samples
    batch["global_attention_mask"][0][0] = 1
    batch["labels"] = outputs.input_ids

    # We have to make sure that the PAD token is ignored
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]

    return batch


def postprocess_text(preds, labels):
	preds = [pred.strip() for pred in preds]
	labels = [label.strip() for label in labels]

	# rougeLSum expects newline after each sentence
	preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
	labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

	return preds, labels


def compute_metrics(pred):
	labels_ids = pred.label_ids
	pred_ids = pred.predictions

	pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
	labels_ids[labels_ids == -100] = tokenizer.pad_token_id
	label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

	# Some simple post-processing
	pred_str, label_str = postprocess_text(pred_str, label_str)

	result = rouge.compute(
		predictions=pred_str, references=label_str, use_stemmer=True
	)

	# Extract a few results from ROUGE
	result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

	prediction_lens = [
		np.count_nonzero(pred != tokenizer.pad_token_id) for pred in pred_ids
	]
	result["gen_len"] = np.mean(prediction_lens)
	result = {k: round(v, 4) for k, v in result.items()}

	return result

model_name = "nsi319/legal-led-base-16384"
tokenizer = AutoTokenizer.from_pretrained(model_name)
exp = 'exp1'
encoder_max_length = 1024*16
decoder_max_length = 1024
batch_size = 1
n_epochs = 3
dataPath = "Summary-Data-IN"
train_df = getData(tokenizer, f'{dataPath}/train-data', encoder_max_length-2)
train_dataset = Dataset.from_pandas(train_df)
val_df = getData(tokenizer, f'{dataPath}/train-data', encoder_max_length-2,1)
val_dataset = Dataset.from_pandas(val_df)

train_dataset = train_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
    remove_columns=["document", "summary"],
)
val_dataset = val_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
    remove_columns=["document", "summary"],
)
# set Python list to PyTorch tensor
train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)

# set Python list to PyTorch tensor
val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)

training_args = Seq2SeqTrainingArguments(
	output_dir=f"results/led/final/{exp}",
	num_train_epochs=n_epochs,
	per_device_train_batch_size=batch_size,
	per_device_eval_batch_size=batch_size,
	fp16=True,
	evaluation_strategy="epoch",
	save_strategy="epoch",
	load_best_model_at_end=True,
	metric_for_best_model="eval_rouge2",
	greater_is_better=True,
	warmup_steps=200,
	predict_with_generate=True,
	logging_dir=f"led_logs/final/{exp}",
	logging_steps=50,
    gradient_accumulation_steps=4,
	save_total_limit=1 #save only the best model
)

# load model + enable gradient checkpointing & disable cache for checkpointing
led = AutoModelForSeq2SeqLM.from_pretrained(model_name, gradient_checkpointing=True, use_cache=False)
# led.resize_token_embeddings(len(tokenizer))

# set generate hyperparameters
led.config.num_beams = 2
led.config.max_length = decoder_max_length
led.config.min_length = 256
# led.config.length_penalty = 2.0
led.config.early_stopping = True
led.config.no_repeat_ngram_size = 4

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=led,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

#Save the finetuned model
# model_checkpoint_dir = f"results/led/{exp}/best_model"
# trainer.save_model(model_checkpoint_dir)

trainer.save_model("./final_model/IN_model")
