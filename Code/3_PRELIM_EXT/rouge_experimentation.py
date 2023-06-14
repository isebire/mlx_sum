# rouge experimentation for classifier

from rouge_score import rouge_scorer
from datasets import load_dataset

scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
mlx = load_dataset("isebire/multi_lexsum_CLEANED_2")

#text = 'I am a fish'
#reference = 'I am a small green fish that swims in a pond. I like it a lot'
#presicion, recall, f_measure = scorer.score(text, reference)['rouge2']
# print(recall)

rouge2_paras = []

for split in ['validation']:

    for case in [mlx[split][0]]:  # classic
        short_summary = case['summary/long_w_chain'].split('[SUMMARY]')[1].strip()

        # Get source documents as a list of paragraphs
        sources = case['sources_clean']
        flat = '\n'.join(sources)
        paras = flat.split('\n')
        # chunks of 5 paras
        para_chunks = ['\n'.join(x) for x in zip(paras[0::5], paras[1::5], paras[2::5], paras[3::5], paras[4::5])]


        # Calculate ROUGE-2 for each paragraph
        for i, para in enumerate(para_chunks):
            # print('** ' + para + '\n\n')
            precision, recall, f_measure = scorer.score(para, short_summary)['rouge2']
            rouge2_paras.append((i, recall))

        # Get top ones
        rouge2_paras = sorted(rouge2_paras, key=lambda x: x[1], reverse=True)
        for i in range(10):
            print(rouge2_paras[i][1])
            input(para_chunks[rouge2_paras[i][0]])



# points just based on not methodological investigation
# works way better with chunks of paragraphs. how many is best? 5 atm
# or - maybe try per paragraph but add also the surrounding paras before and after?
# what about with long summary as reference? i think so but just based on observation
# rouge 1?
