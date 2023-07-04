# Huggingface tutorial https://huggingface.co/docs/transformers/tasks/summarization
# and another at https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization.ipynb#scrollTo=YXTwBEA0MxK1
# the example is for billsum and t5 but trying to adapt as I go to understand :)
# also adapted stuff from other tutorials
# and legal pegasus page https://huggingface.co/nsi319/legal-pegasus

from datasets import load_dataset, DatasetDict, Dataset
from transformers import PegasusTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, pipeline
import evaluate
import numpy as np
import nltk
import torch
import argparse
import pandas

# Need all of this to deal with the filesystem
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True, help="Folder for all the input stuff")
parser.add_argument("--output_dir", type=str, required=True, help="Folder for all the output stuff")


def main(args):
    # MAIN
    # This is going to flatten the source documents and also tokenize
    def preprocess_pegasus(cases):
        MODEL_PATH = f"{args.input_dir}/legal-pegasus"
        tokenizer = PegasusTokenizer.from_pretrained(MODEL_PATH)

        inputs = cases['sources_clean']
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

        labels = tokenizer(text_target=cases["summary/short_w_chain"], max_length=960, truncation=True)    #return_tensors='pt' needed???

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

    # Loading a dataset from csv
    mlx = DatasetDict()
    for split in ['train', 'validation', 'test']:
        split_df = pandas.read_csv(f"{args.input_dir}/mlx-final-{split}.csv")
        split_ds = Dataset.from_pandas(split_df)
        mlx[split] = split_ds
        for case in mlx[split]:
            case['sources_clean'] = case['sources_clean'].join('\n\n')

    MODEL_PATH = f"{args.input_dir}/legal-pegasus"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = PegasusTokenizer.from_pretrained(MODEL_PATH)

    # Preprocess the whole dataset - batched to make it go faster
    tokenized_mlx = mlx.map(preprocess_pegasus, batched=True)

    # Create a  batch of examples using DataCollatorForSeq2Seq
    # more efficient to dynamically pad the sentences to the longest length
    # in a batch during collation, instead of padding the whole dataset to the
    # maximum length.
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)

    # Include a metric during training to evaluate the model's performance
    rouge = evaluate.load("rouge")

    # Define the model
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)  # OR: PegasusForConditionalGeneration

    # Define training hyperparameters
    # only required parameter is output_dir which specifies where to save the model
    # at the end of each epoch, the Trainer will evaluate the ROUGE metric and
    # save the training checkpoint.
    training_args = Seq2SeqTrainingArguments(
        output_dir="legal_pegasus_initial_test",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,   # max number of times it saves the model checkpoints
        num_train_epochs=4,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,   # it won't work otherwise as cluster has no internet
    )

    # Pass the training arguments to Seq2SeqTrainer along with the model,
    # dataset, tokenizer, data collator, and compute_metrics function.
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_mlx["train"],
        eval_dataset=tokenized_mlx["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model!
    trainer.train()

    # Save the model locally as can't access hub on cluster
    trainer.save_model(f"{args.output_dir}/test_trained_pegasus")

    # Now it has been trained, use for inference
    text = mlx['test'][0]['summary/short_w_chain']

    # Using pipeline
    summarizer = pipeline("summarization", model=f"{args.output_dir}/test_trained_pegasus")
    summarizer(text)

    # Alt: manually doing inference
    tokenizer = PegasusTokenizer.from_pretrained(f"{args.output_dir}/test_trained_pegasus")
    input_tokenized = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True, padding="longest").input_ids
    model = AutoModelForSeq2SeqLM.from_pretrained(f"{args.output_dir}/test_trained_pegasus")
    outputs = model.generate(input_tokenized,
                             num_beams=8,
                             no_repeat_ngram_size=3,
                             length_penalty=2.0,
                             min_length=24,
                             max_length=960,
                             early_stopping=True,
                             max_new_tokens=100,
                             do_sample=False)
    summary = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    print("done!")
