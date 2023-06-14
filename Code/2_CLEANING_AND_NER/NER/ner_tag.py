# NER TAGGING

from datasets import load_dataset
import pandas
import pickle
import spacy
import lexnlp.nlp.en.segments.sentences
import itertools
import lexnlp.nlp.en.tokens
import lexnlp.extract.en.money
import nltk
from string import punctuation
import re

PRETRAINED_LABELS = ['DATE', 'PERSON', 'GPE', 'ORG', 'NORP', 'LAW']
SCRATCH_LABELS = ['CLAIMANT_INFO']
LEXNLP_LABELS = ['MONEY']

# Load spacy dependencies
nlp = spacy.load('en_core_web_trf')

# Load NER models
# NER_PT_ROBERTA = spacy.load('/Users/izzy/Desktop/UNI/Diss!/Code/2_CLEANING_AND_NER/NER/pkg_pretrained_roberta/en_pipeline-0.0.0/en_pipeline/en_pipeline-0.0.0')
NER_PT_LEGALBERT = spacy.load('/Users/izzy/Desktop/UNI/Diss!/Code/2_CLEANING_AND_NER/NER/pkg_pretrained_legalbert/en_pipeline-0.0.0/en_pipeline/en_pipeline-0.0.0')
NER_SC_LEGALBERT = spacy.load('/Users/izzy/Desktop/UNI/Diss!/Code/2_CLEANING_AND_NER/NER/pkg_scratch_legalbert/en_pipeline-0.0.0/en_pipeline/en_pipeline-0.0.0')

def get_entities_of_all_types_pretrained(text, model_chosen):

    # Convert text into spacy .Doc format by passing through text pipeline
    text = nlp(text)

    # Run the document through NER tagger
    doc = model_chosen(str(text))

    entity_tuples = []

    entities = [ent for ent in doc.ents if ent.label_ in PRETRAINED_LABELS] or []
    if entities != []:
        for entity in entities:

            # Amazingly extracting the nouns somewhat fixes segmentation issues
            if entity.label_ == 'GPE' or entity.label_ == 'ORG':
                tokens = entity.text.split(' ')
                tags = nltk.pos_tag(tokens)

                # also keep adjectives
                # also keep 'of'
                nouns = [word for word,pos in tags if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS' or pos == 'JJ' or word == 'in' or word == 'of')]

                # need to work out if they are consecutive in original
                try:
                    entity_new = nouns[0]
                    for word in nouns[1:]:
                        if entity_new + ' ' + word in entity.text:
                            entity_new = entity_new + ' ' + word
                        else:
                            entity_tuples.append((entity_new, entity.label_))
                            entity_new = word
                    if entity_new != '':
                        entity_tuples.append((entity_new.strip(punctuation), entity.label_))
                except:
                    pass

            else:
                # Of a type where segmentation is not a thing needed
                entity_tuples.append((entity.text.strip(punctuation), entity.label_))

    return entity_tuples

def get_entities_of_all_types_scratch(text, model_chosen):

    # Convert text into spacy .Doc format by passing through text pipeline
    text = nlp(text)

    # Run the document through NER tagger
    doc = model_chosen(str(text))

    entity_tuples = []

    entities = [ent for ent in doc.ents if ent.label_ in SCRATCH_LABELS] or []
    if entities != []:
        for entity in entities:
            entity_tuples.append((entity.text, entity.label_))

    return entity_tuples


def get_money_lexnlp(text):

    money_list = list(lexnlp.extract.en.money.get_money(text, return_sources=True))
    entities = []
    for item in money_list:
        # input(item)
        # need to trim this too in certain cases - format 'cost .... $..'
        # if it contains a number
        if any(char.isdigit() for char in item[2]):
            words = item[2].split(' ')
            for i, word in enumerate(words):
                if any(char.isdigit() for char in word):
                    stripped = ' '.join(words[i:])
                    break
            entities.append((stripped, 'MONEY'))
        else:
            entities.append((item[2], 'MONEY'))
    return entities


def ner_legalbert_passage_by_sentence(text, verbose=False):

    # can't just segment by '.' or '. ' as abbreviations
    sentences = lexnlp.nlp.en.segments.sentences.get_sentence_list(text)

    # to construct chain, following original paper
    # entities per sentence, and sep for each sentence with |||, each entity with | ?
    # keep each entity only the first time it occurs
    entities_by_sentence = []
    chain = ''
    entities_in_chain = []

    for sentence in sentences:
        pretrained_entities = get_entities_of_all_types_pretrained(sentence, NER_PT_LEGALBERT)
        scratch_entities = get_entities_of_all_types_scratch(sentence, NER_SC_LEGALBERT)
        lexnlp_entities = get_money_lexnlp(sentence)

        all_entities_in_sentence = pretrained_entities + scratch_entities + lexnlp_entities

        all_entities_in_sentence = sorted(all_entities_in_sentence, key=lambda x: sentence.index(x[0]))

        entities_by_sentence.append(all_entities_in_sentence)
        entities_added = False
        for entity, type in all_entities_in_sentence:
            if entity not in entities_in_chain:
                chain = chain + entity + ' | '
                entities_in_chain.append(entity)
                entities_added = True
        if entities_added:
            chain = chain + ' ||| '

    chain = chain + '*'
    chain = chain.replace(' |  ||| ', ' ||| ')
    chain = chain.replace(' ||| *', '')

    if verbose:
        print(chain)

    entities_w_labels = list(itertools.chain.from_iterable(entities_by_sentence))
    entities = [i[0] for i in entities_w_labels]
    labels = [i[1] for i in entities_w_labels]

    return entities_w_labels, entities, labels, chain
