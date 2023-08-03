from transformers import PegasusTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
import torch

names = ['trained_pegasus_2cO', 'trained_pegasus_2d', 'trained_pegasus_3cO', 'trained_pegasus_3d', 'trained_pegasus_4cO', 'trained_pegasus_4d', 'trained_pegasus_5cO', 'trained_pegasus_5d']


for model_name in names:

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.push_to_hub(model_name, private=True)
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model.push_to_hub(model_name, private=True)
