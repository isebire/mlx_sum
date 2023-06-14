# test lexnlp

import lexnlp.extract.en.entities.nltk_re

print('Reading file...')
with open('lexnlp_noisy_doc.txt', 'r') as f:
    doc = f.read()

entities = list(lexnlp.extract.en.entities.nltk_re.get_entities.nltk_re(doc))

#doc = lexnlp.nlp.en.segments.paragraphs(doc)
print(entities)
