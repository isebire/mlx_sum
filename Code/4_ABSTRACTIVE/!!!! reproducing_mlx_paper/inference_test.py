# Testing format for inference!!

from datasets import load_dataset, DatasetDict, Dataset, load_from_disk
from transformers import PegasusTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, pipeline
import evaluate
import numpy as np
import nltk
import torch
import argparse
import pandas
import gc

# Need all of this to deal with the filesystem
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True, help="Folder for all the input stuff")
parser.add_argument("--output_dir", type=str, required=True, help="Folder for all the output stuff")


def main(args):
    # MAIN
    # This is going to flatten the source documents and also tokenize
    def preprocess_pegasus(cases):

        # *** here - model type
        MODEL_PATH = f"{args.input_dir}/pegasus-cnn_dailymail"
        tokenizer = PegasusTokenizer.from_pretrained(MODEL_PATH)

        # *** here - input col
        inputs = cases['first_k_source']
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

        # *** here - target col and length
        labels = tokenizer(text_target=cases["summary/short"], max_length=960, truncation=True)    #return_tensors='pt' needed???

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    # This will compute ROUGE from predictions and labels
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        # use_aggregator = False returns a list, computing a metric for each sentence
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=False)

        # Mean generation length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    # Loading a dataset from hf on disk
    print('* Loading dataset...')
    mlx = DatasetDict()
    for split in ['train', 'validation', 'test']:
        # *** Filename!
        fn = f"{args.input_dir}/mlx_OG_reproducing_" + split + ".hf"
        mlx_split = load_from_disk(fn)
        mlx[split] = mlx_split

    # *** Model name
    MODEL_PATH = f"{args.input_dir}/pegasus-cnn_dailymail"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('! Device is ' + device)

    if device == 'cuda':
        print('* Clearing gpu memory...')
        gc.collect()
        torch.cuda.empty_cache()

    # Alt: manually doing inference

    print('* Loading tokenizer for inference...')
    tokenizer = PegasusTokenizer.from_pretrained(MODEL_PATH)

    print('* Loading trained pegasus model...')
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

    results_df_list = []

    for i, case in enumerate(mlx['test']):
        print('** Analysing case ' + str(i) + '...')

        case_data = {}

        case_data['id'] = case['id']

        # *** target col
        gold = case['summary/short']
        case_data['gold'] = gold

        # *** input col
        text = case['first_k_source']
        case_data['source'] = text

        print('* Tokenizing input....')
        input_tokenized = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True, padding="longest").input_ids



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

        # just for testing :)
        if i > 5:
            break

    results_df = pandas.DataFrame(results_df_list)
    # *** filename
    filename = f"{args.output_dir}/outputs_test.csv"
    results_df.to_csv(filename)

    print(results_df.head())


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    print("done!")
