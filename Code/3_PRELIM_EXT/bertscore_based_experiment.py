# rouge experimentation for classifier

from bert_score import score
from datasets import load_dataset

mlx = load_dataset("isebire/multi_lexsum_CLEANED_2")

text = 'I am a fish'
reference = 'I am a small green fish that swims in a pond. I like it a lot'
P, R, F1 = score([text], [reference], lang="en", verbose=True)
print(R)
'''

bs_paras = []

for split in ['validation']:

    for case in [mlx[split][0]]:  # classic
        summary = case['summary/long_w_chain'].split('[SUMMARY]')[1].strip()

        # Get source documents as a list of paragraphs
        sources = case['sources_clean']
        flat = '\n'.join(sources)
        paras = flat.split('\n')
        # chunks of 5 paras
        # para_chunks = ['\n'.join(x) for x in zip(paras[0::5], paras[1::5], paras[2::5], paras[3::5], paras[4::5])]

        # Calculate ROUGE-2 for each paragraph
        for i, para in enumerate(paras):
            # print('** ' + para + '\n\n')
            P, R, F1 = score([para], [summary], lang="en", verbose=True)
            bs_paras.append((i, R))

        # Get top ones
        bs_paras = sorted(bs_paras, key=lambda x: x[1], reverse=True)
        for i in range(10):
            print(bs_paras[i][1])
            input(paras[bs_paras[i][0]])
'''

# points just based on not methodological investigation
# works way better with chunks of paragraphs. how many is best? 5 atm
# or - maybe try per paragraph but add also the surrounding paras before and after?
# what about with long summary as reference? i think so but just based on observation
# rouge 1?
