# 1B: First K, CNN/DM Pegasus, No entity chaining

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
    # MAIN
    # This is going to flatten the source documents and also tokenize
    def preprocess_pegasus(cases):

        # *** here - model type: CNN/DAILYMAIL
        MODEL_PATH = f"{args.input_dir}/pegasus-cnn_dailymail"
        tokenizer = PegasusTokenizer.from_pretrained(MODEL_PATH)

        # *** here - input col
        inputs = cases['first_k']
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

        # *** here - target col and length  (960 w/o chain, just set to max 1024 with cahin)
        labels = tokenizer(text_target=cases["summary/short_no_chain"], max_length=960, truncation=True)    #return_tensors='pt' needed???

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
        fn = f"{args.input_dir}/mlx_PEGASUS_" + split + ".hf"
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

    print('* Loading tokenizer...')
    tokenizer = PegasusTokenizer.from_pretrained(MODEL_PATH)

    # Preprocess the whole dataset - batched to make it go faster
    print('* Preprocessing dataset...')
    tokenized_mlx = mlx.map(preprocess_pegasus, batched=True)

    # Create a  batch of examples using DataCollatorForSeq2Seq
    # more efficient to dynamically pad the sentences to the longest length
    # in a batch during collation, instead of padding the whole dataset to the
    # maximum length.
    print('* Making batches...')
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=MODEL_PATH)

    # Include a metric during training to evaluate the model's performance
    print('* Loading rouge metric...')
    rouge = evaluate.load("rouge")

    if device == 'cuda':
        print('* Clearing gpu memory...')
        gc.collect()
        torch.cuda.empty_cache()

    # Define the model
    print('* Loading model and sending to cuda')
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)  # OR: PegasusForConditionalGeneration

    # Define training hyperparameters
    # only required parameter is output_dir which specifies where to save the model
    # at the end of each epoch, the Trainer will evaluate the ROUGE metric and
    # save the training checkpoint.
    print('* Setting up training args...')
    training_args = Seq2SeqTrainingArguments(
        # *** Output dir
        output_dir=f"{args.output_dir}/" + "pegasus_1b",
        evaluation_strategy="epoch",
        learning_rate=5e-5, # to match mlx paper. in original tutorial: 2e-5
        per_device_train_batch_size=4,  # was 16 for both but memory error
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=64,   # 256/4=64
        gradient_checkpointing=True,  # memory help
        weight_decay=0.01,
        save_total_limit=3,   # max number of times it saves the model checkpoints
        num_train_epochs=6,  # to match mlx paper. in original tutorial: 4
        predict_with_generate=True,
        fp16=True,   # was originally commented out?
        push_to_hub=False,   # it won't work otherwise as cluster has no internet
    )

    # Pass the training arguments to Seq2SeqTrainer along with the model,
    # dataset, tokenizer, data collator, and compute_metrics function.
    print('* Passing args to trainer...')
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_mlx["train"],
        eval_dataset=tokenized_mlx["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
        # original: compute_metrics=compute_metrics,
        # changed as suggestion to stop cuda issue
    )

    # Train the model!
    print('* Train the model!! (this will take a long time)')
    trainer.train()

    # Save the model locally as can't access hub on cluster
    print('* Saving trained model locally')
    # *** Model name
    output_model_fn = 'trained_pegasus_1b'
    output_model_path = f"{args.output_dir}/" + output_model_fn
    trainer.save_model(output_model_path)

    # Now it has been trained, use for inference

    # *** Column name
    text = mlx['test'][0]['summary/short_no_chain']

    # Using pipeline
    # summarizer = pipeline("summarization", model=f"{args.output_dir}/test_trained_pegasus")
    # summarizer(text)

    # Alt: manually doing inference

    print('* Loading tokenizer for inference...')
    tokenizer = PegasusTokenizer.from_pretrained(output_model_path)

    print('* Loading trained pegasus model...')
    model = AutoModelForSeq2SeqLM.from_pretrained(output_model_path)

    results_df_list = []

    for i, case in enumerate(mlx['test']):
        print('** Analysing case ' + str(i) + '...')

        case_data = {}

        case_data['id'] = case['id']

        # *** target col
        gold = case['summary/short_no_chain']
        case_data['gold'] = gold

        # *** input col
        text = case['first_k']
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

    results_df = pandas.DataFrame(results_df_list)
    # *** filename
    filename = f"{args.output_dir}/outputs_pegasus_1b.csv"
    results_df.to_csv(filename)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    print("done!")
