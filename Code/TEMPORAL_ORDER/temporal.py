# Reorder MLX-CLEAN with documents in temporal order
from datasets import load_dataset, Dataset, DatasetDict
import pandas
import requests
import re
from bs4 import BeautifulSoup
import time
from dateutil.parser import parse

mlx_clean = load_dataset('isebire/mlx_CLEAN_FINAL')
mlx_meta = load_dataset("allenai/multi_lexsum", name="v20230518")

mlx_temporal = DatasetDict()

for split in ['validation', 'test', 'train']:

    split_data = []

    for i, case in enumerate(mlx_clean[split]):
        case_data = {}  # this will hold the row of the dataset for this case
        print('** Case ' + str(i) + ' of ' + split)

        # Match to case in metadata ver by id
        for original_case in mlx_meta[split]:
            if original_case['id'] == case['id']:
                break

        # Retrieve the links
        doc_links = original_case['sources_metadata']['url']
        doc_types = original_case['sources_metadata']['doc_type']

        docket_temporal_order = []

        # Get the temporal order
        temporal_order = []
        for original_position, url in enumerate(doc_links):



            if url is None:
                # very rare special case
                if doc_types[original_position] == 'Docket':
                    docket_temporal_order.append((original_position, parse('01/01/2099')))
                else:
                    temporal_order.append((original_position, parse('01/01/2099')))
                continue


            # UA to stop access denied when request sent
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

            complete = False
            while complete is False:
                try:
                    r = requests.get(url, headers=headers)
                    complete = True
                except:
                    print('Error - waiting')
                    time.sleep(30)

            while r.status_code != 200:
                print('ERROR')
                print(r.status_code)
                print('Waiting...')
                time.sleep(60)
                print('Trying again....')
                r = requests.get(url, headers=headers)
            soup = BeautifulSoup(r.content, features="lxml")
            div = soup.find("div", {"class": "flex-grow-1"})
            # more complicated than originally as may be multiple sets of brackets
            # and date is in last one
            try:
                date = str(div).split("<h1", 1)[1].split('</h1')[0].split(')')[-2].split('(')[-1]
                print(date)
                if doc_types[original_position] == 'Docket':
                    docket_temporal_order.append((original_position, parse(date)))
                else:
                    temporal_order.append((original_position, parse(date)))

            except:
                # some documents (eg associated forms) actually have no date - put at end
                if doc_types[original_position] == 'Docket':
                    docket_temporal_order.append((original_position, parse('01/01/2099')))
                else:
                    temporal_order.append((original_position, parse('01/01/2099')))
                print('! No date - putting at end')
                print(url)
            #time.sleep(1)

        temporal_order = sorted(temporal_order, key=lambda x: x[1])
        temporal_indices = [x[0] for x in temporal_order]
        # docket last
        docket_temporal_order = sorted(docket_temporal_order, key=lambda x: x[1])
        docket_temporal_indices = [x[0] for x in docket_temporal_order]
        temporal_indices = temporal_indices + docket_temporal_indices

        # print(temporal_indices)  ###

        # Make new documents list
        print('Reordering....')
        docs = case['sources_clean']  #.split('\n[DOCSPLIT]\n')

        new_docs = []
        for idx in temporal_indices:
            new_docs.append(docs[idx])

        # for doc in new_docs:
        #     print(doc[:500])
        #     print('####')
        # print('----')
        #
        # input('fish')

        case_data['sources_clean'] = new_docs

        # id, summary/long_w_chain, summary/short_w_chain, summary/tiny_w_chain same as prev ver
        case_data['id'] = case['id']
        case_data['summary/long_w_chain'] = case['summary/long_w_chain']
        case_data['summary/short_w_chain'] = case['summary/short_w_chain']
        case_data['summary/tiny_w_chain'] = case['summary/tiny_w_chain']

        split_data.append(case_data)


    # SAVE SPLIT
    # Make the list of dicts into a pandas dataframe
    df = pandas.DataFrame.from_dict(split_data)

    # Make the pandas dataframe into a hugging face dataset
    ds = Dataset.from_pandas(df)

    mlx_temporal[split] = ds

    # Export to csv -> maybe source of list of lists issues
    #filename = '230712_mlx_cleaned_ordered_' + split + '.csv'
    #ds.to_csv(filename, index=None)

    # Export to disk
    filename = '230714_mlx_cleaned_ordered_' + split + '.hf'
    ds.save_to_disk(filename)


# output the DatasetDict huggingface
print('*** Uploading to huggingface')
mlx_temporal.push_to_hub("isebire/230714_mlx_CLEANED_ORDERED", private=True)
