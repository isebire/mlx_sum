# textrank?

from datasets import load_dataset
import spacy
import pytextrank
from icecream import ic

mlx = load_dataset("isebire/multi_lexsum_CLEANED_2")
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")

bs_paras = []

for split in ['validation']:

    for case in [mlx[split][0]]:  # classic
        summary = case['summary/long_w_chain'].split('[SUMMARY]')[1].strip()

        # Get source documents as a list of paragraphs
        sources = case['sources_clean']
        flat = '\n'.join(sources)

        doc = nlp(flat)
        for phrase in doc._.phrases:
            print(phrase.text)
            print(phrase.rank, phrase.count)
            print(phrase.chunks)

        tr = doc._.textrank
        # top ranked phrases
        for phrase in doc._.phrases:
            ic(phrase.rank, phrase.count, phrase.text)
            ic(phrase.chunks)

        print('Summary')

        for sent in tr.summary(limit_phrases=15, limit_sentences=10):
            ic(sent)
