# Make a new dataset with the cleaned sources that is NER tagged
# For each of the 3 splits (3177, 454, 908 rows)
# Cols:
# id, sources_clean, source_entities, summary/long_w_chain, summary/short_w_chain, summary/tiny_w_chain

print('Running!')

from NER import ner_tag as ner
from NER import entity_stats as entity_stats
import clean
from datasets import load_dataset, Dataset, DatasetDict
import datasets
import pandas
import statistics
import lexnlp.nlp.en.segments.sentences

# Load dataset
mlx = load_dataset("allenai/multi_lexsum", name="v20220616")

def build_entity_types_list(counts_dict, labels):
    for label in labels:
        counts_dict[label] = counts_dict.get(label, 0) + 1

# Initialise the dataset dict that we will build by adding each split
mlx_clean = DatasetDict()

# Intialise for stats
#source_entity_types_counts = {}
long_entity_types_counts = {}
short_entity_types_counts = {}
tiny_entity_types_counts = {}

#source_entity_density_list = []
long_entity_density_list = []
short_entity_density_list = []
tiny_entity_density_list = []

long_entity_verified_list = []
short_entity_verified_list = []
tiny_entity_verified_list = []


for split in ['train', 'validation', 'test']:
    print('*** Now analysing split: ' + split)

    # We are going to make a huggingface Dataset for each of these,
    # by making a pandas dataframe first

    split_data = []   # [{'sources_clean': [], etc}]  # list of dicts

    new_short_summaries_in_split = 0
    short_summaries_in_split = 0
    existing_short_summaries_in_split = 0

    # Process the data for each case
    cases_in_split = len(mlx[split])
    for i, case in enumerate(mlx[split]):
    #for case in [mlx[split][5]]:    <- for testing
        print('** Now analysing case ' + str(i + 1) + ' of ' + str(cases_in_split) + ' in split')

        case_data = {}  # this will hold the row of the dataset for this case

        case_data['id'] = case['id']

        # Clean each source document and store in a list, like before
        print('* Cleaning source documents')
        cleaned_doc_list = [clean.clean_document(i) for i in case['sources']]
        source_text = ''
        for doc in cleaned_doc_list:
            source_text = source_text + '\n\n' + doc

        case_data['sources_clean'] = cleaned_doc_list

        ## Removed for efficiency reasons as not strictly needed
        # print('* Tagging source documents (removable)')
        # Tagging cleaned data -> remove?
        # source_entities = []
        # for doc in cleaned_doc_list:
        #     entities_w_labels, entities, labels, chain = ner.ner_legalbert_passage_by_sentence(doc)
        #     source_entities.append(entities)
        #
        #     build_entity_types_list(source_entity_types_counts, labels)
        #     source_entity_density_list.append(entity_stats.entity_density(entities, doc))
        # source_entities = [item for sublist in source_entities for item in sublist]


        bad_case = False
        for sum_len in ['long', 'short', 'tiny']:
            print('* Considering summary length: ' + sum_len)
            if case['summary/' + sum_len] is not None:
                summary = case['summary/' + sum_len]

                # Special case, remove sentence as often contains "hallucination"
                sentences = lexnlp.nlp.en.segments.sentences.get_sentence_list(summary)
                if 'as of' in sentences[-1]:
                    # Remove last sentence
                    sentences = sentences[:-1]
                    summary = ' '.join(sentences)


                # Entity tags the document
                print('* NER tagging')
                entities_w_labels, entities, labels, chain = ner.ner_legalbert_passage_by_sentence(summary)

                # Calculate stats
                # in some cases eg tiny summary, there may be no entities found
                if len(entities_w_labels) > 0:
                    #print('Entity types found')
                    #print(labels)
                    entity_density = entity_stats.entity_density(entities, summary)
                    #print('Entity density')
                    #print(entity_density)
                    entities_verified_percentage = entity_stats.entities_verified(source_text, entities_w_labels)
                    #print('Entities verfified')
                    #print(entities_verified_percentage)

                    if entities_verified_percentage < 0.75:  # based on inspection
                        bad_case = True
                        print('!! Likely OCR errors')
                        break

                    # Save the stats for aggregrate stats later
                    if sum_len == 'long':
                        build_entity_types_list(long_entity_types_counts, labels)
                        long_entity_density_list.append(entity_density)
                        long_entity_verified_list.append(entities_verified_percentage)

                    elif sum_len == 'short':
                        build_entity_types_list(short_entity_types_counts, labels)
                        short_entity_density_list.append(entity_density)
                        short_entity_verified_list.append(entities_verified_percentage)

                    elif sum_len == 'tiny':
                        build_entity_types_list(tiny_entity_types_counts, labels)
                        tiny_entity_density_list.append(entity_density)
                        tiny_entity_verified_list.append(entities_verified_percentage)

                # DO THIS!
                # could add some processing here to remove things from
                # summary where entities don't match? -> see if needed based on data

                if bad_case is True:
                    break

                # print('Constructing entity chain')

                summary_w_chain = '[ENTITYCHAIN] ' + chain + ' [SUMMARY] ' + summary

                case_data['summary/' + sum_len + '_w_chain'] = summary_w_chain
                existing_short_summaries_in_split += 1
                short_summaries_in_split += 1

            else:
                if sum_len == 'short' and bad_case is False:
                    long_summary_words = ' '.join(case['summary/long'].replace('\n\n', ' ').split()).split(' ')
                    num_summary_words = len(long_summary_words)

                    if num_summary_words <= 671: # the max words in a short summary
                        print('! The long summary is actually short summary length (' + str(num_summary_words) + ' words) so adding!')
                        case_data['summary/short_w_chain'] = case_data['summary/long_w_chain']
                        new_short_summaries_in_split += 1
                        short_summaries_in_split += 1
                        continue

                case_data['summary/' + sum_len + '_w_chain'] = None


        # Now the dict should have info for the following columns
        # id
        # sources_clean, source_entities
        # summary/long_w_chain, summary/short_w_chain, summary/tiny_w_chain

        if bad_case is False:
            split_data.append(case_data)

        # input('Next case?')


    print('** Making dataset for split')

    # Make the list of dicts into a pandas dataframe
    df = pandas.DataFrame.from_dict(split_data)

    # Make the pandas dataframe into a hugging face dataset
    ds = Dataset.from_pandas(df)

    # Add this to the overall DatasetDict
    mlx_clean[split] = ds

    # Print stats
    # NOTE: THIS STATS WERE CALCULATED WRONG, BUT RECTIFIED IN A LATER SCRIPT
    print('***** INFO')
    print('for ' + split)
    print('exisiting short summaries: ' + str(existing_short_summaries_in_split))
    print('new short summaries: ' + str(new_short_summaries_in_split))
    print('overall short summaries: ' + str(short_summaries_in_split))

    # Export to csv just in case
    filename = 'mlx_clean_' + split + '.csv'
    ds.to_csv(filename, index = None)

# Want to calculate proportion of entities of each type as an aggregrate stat
# across all cases in all splits here!
# for cleaned source docs, long, short, and tiny

print('*** Stats across all splits!')

#print(source_entity_types_counts)
#entity_stats.entity_types(source_entity_types_counts, 'source_ent_types')
print(long_entity_types_counts)
entity_stats.entity_types(long_entity_types_counts, 'long_ent_types')
print(short_entity_types_counts)
entity_stats.entity_types(short_entity_types_counts, 'short_ent_types')
print(tiny_entity_types_counts)
entity_stats.entity_types(tiny_entity_types_counts, 'tiny_ent_types')

# Output also average entity density
# for cleaned source docs, long, short, and tiny
#print('mean source entity density')
#print(statistics.mean(source_entity_density_list))
print('mean long entity density')
print(statistics.mean(long_entity_density_list))
print('mean short entity density')
print(statistics.mean(short_entity_density_list))
print('mean tiny entity density')
print(statistics.mean(tiny_entity_density_list))


# Output also average / min / max % of entities in source
# for long, short, and tiny
print('entity hallucination - long. max mean min')
print(max(long_entity_verified_list))
print(statistics.mean(long_entity_verified_list))
print(min(long_entity_verified_list))

print('entity hallucination - short. max mean min')
print(max(short_entity_verified_list))
print(statistics.mean(short_entity_verified_list))
print(min(short_entity_verified_list))

print('entity hallucination - tiny. max mean min')
print(max(tiny_entity_verified_list))
print(statistics.mean(tiny_entity_verified_list))
print(min(tiny_entity_verified_list))


# output the DatasetDict huggingface
print('*** Uploading to huggingface')
mlx_clean.push_to_hub("isebire/multi_lexsum_CLEANED", private=True)
