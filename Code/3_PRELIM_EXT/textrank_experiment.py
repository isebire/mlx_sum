# textrank?

from datasets import load_dataset
import spacy
import pytextrank
from icecream import ic

mlx = load_dataset("isebire/230714_mlx_CLEANED_ORDERED")
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")

bs_paras = []

for split in ['test']:

    for case in mlx[split]:  # classic
        summary = case['summary/short_w_chain'].split('[SUMMARY]')[1]

        # Get source documents as a list of paragraphs
        sources = case['sources_clean']
        sources_no_docket = []
        for source in sources:
            if not 'CIVIL DOCKET' in doc:
                sources_no_docket.append(source)
        flat = '\n'.join(sources_no_docket)

        doc = nlp(flat)
        tr = doc._.textrank

        print('Summary')

        summary_sents = tr.summary(limit_sentences=30)   # limit_phrases=15

        # Order
        summary_sents = sorted([str(x) for x in summary_sents], key=lambda x: flat.index(x))

        tr_summary = ''.join(summary_sents).replace('\n', ' ')
        print(tr_summary)

        input('fish')
