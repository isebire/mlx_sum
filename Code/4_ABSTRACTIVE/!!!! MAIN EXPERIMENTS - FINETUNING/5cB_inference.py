
from datasets import load_dataset, DatasetDict, Dataset, load_from_disk
from transformers import PegasusTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, pipeline
import evaluate
import numpy as np
import nltk
import torch
import argparse
import pandas
import gc
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Need all of this to deal with the filesystem
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True, help="Folder for all the input stuff")
parser.add_argument("--output_dir", type=str, required=True, help="Folder for all the output stuff")


def main(args):
    # Loading a dataset from hf on disk
    print('* Loading dataset...')
    mlx = DatasetDict()
    for split in ['train', 'validation', 'test']:
        # *** Filename!
        fn = f"{args.input_dir}/mlx_PEGASUS_" + split + ".hf"
        mlx_split = load_from_disk(fn)
        mlx[split] = mlx_split

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # *** Alt: manually doing inference
    output_model_path = f"{args.output_dir}/trained_pegasus_5cO"

    print('* Loading tokenizer for inference...')
    tokenizer = PegasusTokenizer.from_pretrained(output_model_path)

    print('* Loading trained pegasus model...')
    model = AutoModelForSeq2SeqLM.from_pretrained(output_model_path).to(device)

    results_df_list = []

    for i, case in enumerate(mlx['test']):
        print('** Analysing case ' + str(i) + '...')

        case_data = {}

        case_data['id'] = case['id']

        # *** target col
        gold = case['summary/short_chain_both']
        case_data['gold'] = gold

        # *** input col
        text = case['bert_sents']
        case_data['bert_source'] = text

        print('* Tokenizing input....')
        input_tokenized = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True, padding="longest").input_ids
        input_tokenized = input_tokenized.to(device)

        print('* Performing inference!')
        outputs = model.generate(input_tokenized,
                                 num_beams=5,   # to match mlx paper. in original tutorial: 8
                                 no_repeat_ngram_size=3,  # mlx paper: ngram repetition blocks n>3
                                 length_penalty=2.0,  # > 0 so encourages long sequences
                                 min_length=24,  # based on data, without entity chain (34 with)
                                 max_length=960,  # based on data, without entity chain (1154 with)
                                 early_stopping=True,
                                 # max_new_tokens=100,
                                 do_sample=False)
        summary = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        case_data['pegasus_output'] = summary

        results_df_list.append(case_data)

    results_df = pandas.DataFrame(results_df_list)
    # *** filename
    filename = f"{args.output_dir}/outputs_pegasus_5cB.csv"
    results_df.to_csv(filename)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    print("done!")
