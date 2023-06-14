# Pegasus test
# Adapted from example for inference from https://huggingface.co/nsi319/legal-pegasus

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PegasusTokenizer
import torch

mlx = load_dataset('isebire/multi_lexsum_CLEANED_AGAIN')

model_name = "nsi319/legal-pegasus"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)  # OR: PegasusForConditionalGeneration

for split in ['validation']:

    for case in [mlx[split][0]]:  # classic
        short_summary_no_chain = case['summary/short_w_chain'].split('[SUMMARY]')[1].strip()

        # Get source documents as a list of paragraphs
        sources = case['sources_clean']
        text = '\n'.join(sources)

        # note: 'maximum length of input sequence is 1024 tokens'
        input_tokenized = tokenizer.encode(text, return_tensors='pt',max_length=1024,truncation=True)
        summary_ids = model.generate(input_tokenized,
                                          num_beams=9,
                                          no_repeat_ngram_size=3,
                                          length_penalty=2.0,
                                          min_length=24,
                                          max_length=960,
                                          early_stopping=True)
        summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]





# General pegasus example is like this
batch = tokenizer(src_text, truncation=True, padding="longest", return_tensors="pt").to(device)
translated = model.generate(**batch)
tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)

# Another summarisation eg
inputs = tokenizer(ARTICLE_TO_SUMMARIZE, max_length=1024, return_tensors="pt")
summary_ids = model.generate(inputs["input_ids"])
tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]



# Info
