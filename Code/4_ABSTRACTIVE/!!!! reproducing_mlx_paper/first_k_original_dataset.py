# Preparing ORIGINAL dataset to try reproduce their results

import pandas
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PegasusTokenizer
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

mlx = load_dataset("allenai/multi_lexsum", name="v20220616")

# Load utils
print('** Loading pegasus tokenizer...')
tokenizer = PegasusTokenizer.from_pretrained("nsi319/legal-pegasus")


# Setup data structures
print('** Setting up data structures...')
mlx_original_reproduction = DatasetDict()
splits = ['train', 'validation', 'test']

for split in splits:
    split_data = []
    for i, case in enumerate(mlx[split]):
        case_data = {}  # this will hold the row of the dataset for this case
        print('** Case ' + str(i) + ' of ' + split)

        if case['summary/short'] is not None:

            case_data['id'] = case['id']
            case_data['summary/short'] = case['summary/short']


            sources = case['sources']
            # Process the input as first k
            number_documents = len(sources)
            tokens_per_document = int(1024 / number_documents)
            first_k_string = ''
            for document in sources:
                document_tokens = tokenizer.encode(document)[:tokens_per_document]
                document_string = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(document_tokens))
                first_k_string = first_k_string + document_string
            case_data['first_k_source'] = first_k_string

            split_data.append(case_data)

    # SAVE SPLIT
    # Make the list of dicts into a pandas dataframe
    df = pandas.DataFrame.from_dict(split_data)

    # Make the pandas dataframe into a hugging face dataset
    ds = Dataset.from_pandas(df)

    mlx_original_reproduction[split] = ds

    # Export to disk
    filename = 'mlx_OG_reproducing_' + split + '.hf'
    ds.save_to_disk(filename)


# output the DatasetDict huggingface
print('*** Uploading to huggingface')
mlx_original_reproduction.push_to_hub("isebire/mlx_OG_reproducing", private=True)
