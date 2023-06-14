# INVESTIGATING NER

from datasets import load_dataset
import pandas
import pickle
import spacy
import lexnlp.nlp.en.segments.sentences
import itertools

# For lexnlp NER
import lexnlp.nlp.en.tokens
import lexnlp.extract.en.money
import lexnlp.extract.en.acts
import lexnlp.extract.en.amounts
import lexnlp.extract.en.dates
import lexnlp.extract.en.durations
import lexnlp.extract.en.distances
import lexnlp.extract.en.money
import lexnlp.extract.en.percents
import lexnlp.extract.en.pii
import lexnlp.extract.en.ratios
import lexnlp.extract.en.regulations
import lexnlp.extract.en.urls
from lexnlp.extract.en.entities.nltk_maxent import get_company_annotations, get_geopolitical, get_companies, get_persons
import nltk

PRETRAINED_LABELS = ['DATE', 'PERSON', 'GPE', 'ORG', 'NORP', 'LAW']
SCRATCH_LABELS = ['CLAIMANT_EVENT', 'CLAIMANT_INFO', 'PROCEDURE', 'CREDIBILITY', 'DETERMINATION', 'DOC_EVIDENCE', 'EXPLANATION', 'LAW_CASE', 'LAW_REPORT']
LEXNLP_LABELS = ['ACT', 'AMOUNTS', 'DATE', 'DURATION', 'DISTANCE', 'PERCENT', 'MONEY', 'RATIO', 'REGULATION', 'URL', 'COMPANY', 'GPE', 'PERSON']

# Load dataset
#multi_lexsum = load_dataset("allenai/multi_lexsum", name="v20220616")

# Load spacy dependencies
nlp = spacy.load('en_core_web_trf')

'''
# Load NER models
#NER_PT_ROBERTA = spacy.load('./pkg_pretrained_roberta/en_pipeline-0.0.0/en_pipeline/en_pipeline-0.0.0')
#NER_PT_LEGALBERT = spacy.load('./pkg_pretrained_legalbert/en_pipeline-0.0.0/en_pipeline/en_pipeline-0.0.0')
#NER_SC_LEGALBERT = spacy.load('./pkg_scratch_legalbert/en_pipeline-0.0.0/en_pipeline/en_pipeline-0.0.0')

# from Claire
def get_entities_of_type(text, model_chosen, target_ent=''):  # target ent eg GPE

    # Convert text into spacy .Doc format by passing through text pipeline
    text = nlp(text)

    # Run the document through NER tagger
    doc = model_chosen(str(text))

    # Get the entities of the chosen type
    entities = [ent.text for ent in doc.ents if ent.label_ == target_ent] or [""]

    return entities

def get_entities_of_all_types_pretrained(text, model_chosen):  # target ent eg GPE

    # Convert text into spacy .Doc format by passing through text pipeline
    text = nlp(text)

    # Run the document through NER tagger
    doc = model_chosen(str(text))

    entity_tuples = []

    for target_ent in PRETRAINED_LABELS:
        # Get the entities of the chosen type
        entities = [ent.text for ent in doc.ents if ent.label_ == target_ent] or [""]
        if entities != [""]:
            for entity in entities:
                entity_tuples.append((entity, target_ent))

    return entity_tuples

def get_entities_of_all_types_scratch(text, model_chosen):  # target ent eg GPE

    # Convert text into spacy .Doc format by passing through text pipeline
    text = nlp(text)

    # Run the document through NER tagger
    doc = model_chosen(str(text))

    entity_tuples = []

    for target_ent in SCRATCH_LABELS:
        # Get the entities of the chosen type
        entities = [ent.text for ent in doc.ents if ent.label_ == target_ent] or [""]
        if entities != [""]:
            for entity in entities:
                entity_tuples.append((entity, target_ent))

    return entity_tuples


def ner_legalbert_passage_whole(text):

    pretrained_entities = get_entities_of_all_types_pretrained(summary, NER_PT_LEGALBERT)
    scratch_entities = get_entities_of_all_types_scratch(summary, NER_SC_LEGALBERT)

    all_entities = pretrained_entities + scratch_entities

    return all_entities


def ner_legalbert_passage_by_sentence(text):

    # can't just segment by '.' or '. ' as abbreviations
    sentences = lexnlp.nlp.en.segments.sentences.get_sentence_list(text)
    #print('I think the sentences are: ')
    #for sentence in sentences:
    #    print('** ' + sentence)

    entities_by_sentence = []

    for sentence in sentences:
        pretrained_entities = get_entities_of_all_types_pretrained(sentence, NER_PT_ROBERTA)
        scratch_entities = get_entities_of_all_types_scratch(sentence, NER_SC_LEGALBERT)
        lexnlp_entities = get_entities_lexnlp(sentence, target_ent='MONEY')

        all_entities_in_sentence = pretrained_entities + scratch_entities + lexnlp_entities

        entities_by_sentence.append(all_entities_in_sentence)

    return entities_by_sentence
'''

def get_entities_lexnlp(text, target_ent=''):

    if target_ent == 'ACT':
        return list(lexnlp.extract.en.acts.get_act_list(text))

    if target_ent == 'AMOUNT':
        return list(lexnlp.extract.en.amounts.get_amounts(text))

    if target_ent == 'DATE':
        return list(lexnlp.extract.en.dates.get_dates(text))

    if target_ent == 'DURATION':
        return list(lexnlp.extract.en.durations.get_durations(text))

    if target_ent == 'DISTANCE':
        return list(lexnlp.extract.en.distances.get_distances(text))

    if target_ent == 'PERCENT':
        return list(lexnlp.extract.en.percents.get_percents(text))

    if target_ent == 'MONEY':
        money_list = list(lexnlp.extract.en.money.get_money(text))
        print(money_list)
        entities = []
        for item in money_list:
            entities.append((item[1] + ' ' + str(item[0]), 'MONEY'))
        return entities

    if target_ent == 'RATIO':
        return list(lexnlp.extract.en.ratios.get_ratios(text))

    if target_ent == 'REGULATION':
        return list(lexnlp.extract.en.regulations.get_regulations(text))

    if target_ent == 'URL':
        return list(lexnlp.extract.en.urls.get_urls(text))

    if target_ent == 'COMPANY':
        companies = list(get_company_annotations(text))[::2]
        return companies

    if target_ent == 'GPE':
        return list(get_geopolitical(text))

    if target_ent == 'PERSON':
        return list(get_persons(text))


x = 'AT&T employee against AT&T Corp.'
tokens = nltk.word_tokenize(x)
tokens = x.split(' ')
tags = nltk.pos_tag(tokens)
nouns = [word for word,pos in tags if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]
print(nouns)

# need to work out if they are consecutive in original
entity_new = nouns[0]
for word in nouns[1:]:
    if entity_new + ' ' + word in x:
        entity_new = entity_new + ' ' + word
    else:
        print('found entity')
        print(entity_new)
        entity_new = word
if entity_new != '':
    print('found entity')
    print(entity_new)


'''
# REDUCED FOR TESTING
for split in ['validation']:

    for case in [multi_lexsum[split][0]]:  # just the classic case

        docs = case['sources']
        for sum_len in ['long']:
            summary = case["summary/" + sum_len]

            if summary is None:
                continue

            print('---------------------------')
            print('Length: ' + sum_len)
            print(summary)

            entities = ner_legalbert_passage_by_sentence(summary)
            for e in entities:
                print('***')
                print(e)
            entities_w_labels = list(itertools.chain.from_iterable(entities))
            entities = [i[0] for i in entities_w_labels]
            labels = [i[1] for i in entities_w_labels]

            print(entities)
            print(labels)
'''
