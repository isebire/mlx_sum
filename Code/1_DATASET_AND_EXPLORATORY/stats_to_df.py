# length analysis

# -	max / min / mean / dist – number of documents per case
# -	max / min / mean / dist  - number of words / tokens per case
# -	max / min / mean / dist – number of words / tokens per document
# -	max / min / mean / dist  - number of words / tokens per each 3 types of summary
# -	compression ratio |source|/ |summary| for all summary lengths and long → short


from datasets import load_dataset
import pandas
import pickle
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

multi_lexsum = load_dataset("allenai/multi_lexsum", name="v20220616")

DATA_FOR_STATS_DF = []
DOCUMENT_LENGTHS_LIST = []

for split in ['train', 'validation', 'test']:
    print('Documents in ' + split)
    print(len(multi_lexsum[split]))

    for case in multi_lexsum[split]:
        print('fish')

        case_data = {}
        docs = case['sources']
        case_data['number_documents'] = len(docs)

        total = 0
        for doc in docs:
            words_in_doc = len(' '.join(doc.replace('\n\n', ' ').split()).split(' '))
            total += words_in_doc
            DOCUMENT_LENGTHS_LIST.append(words_in_doc)

        case_data['total_words_documents'] = total

        concat_source = ' '.join(docs)
        source_words = ' '.join(concat_source.replace('\n\n', ' ').split()).split(' ')

        for sum_len in ["long", "short", "tiny"]:
            summary = case["summary/" + sum_len]
            if summary is not None:

                # looks weird but so it calculates correctly with newlines etc
                summary_words = ' '.join(summary.replace('\n\n', ' ').split()).split(' ')
                num_summary_words = len(summary_words)

                summary_words_in_source = 0
                for word in summary_words:
                    if word in source_words:
                        summary_words_in_source += 1

                index = sum_len + '_words'
                case_data[index] = num_summary_words

                index = sum_len + '_extractive_words_ratio'
                case_data[index] = summary_words_in_source / num_summary_words

                index = sum_len + '_compression_ratio'
                case_data[index] = total / num_summary_words

                long_words = ' '.join(case['summary/long'].replace('\n\n', ' ').split()).split(' ')
                if sum_len == 'short':
                    short_words_in_long = 0
                    for word in summary_words:
                        if word in long_words:
                            short_words_in_long += 1

                    case_data['long_to_short_extractive_words_ratio'] = short_words_in_long / num_summary_words
                    case_data['long_to_short_compression_ratio'] = case_data['long_words'] / num_summary_words

            else:
                index = sum_len + '_words'
                case_data[index] = None

                index = sum_len + '_extractive_words_ratio'
                case_data[index] = None

                index = sum_len + '_compression_ratio'
                case_data[index] = None

                if sum_len == 'short':

                    case_data['long_to_short_extractive_words_ratio'] = None
                    case_data['long_to_short_compression_ratio'] = None


        DATA_FOR_STATS_DF.append(case_data)

STATS_DF = pandas.DataFrame.from_dict(DATA_FOR_STATS_DF)
STATS_DF.to_csv('exploratory_stats.csv', index=False)

with open('document_lengths.pkl', 'wb') as f:
    pickle.dump(DOCUMENT_LENGTHS_LIST, f)
