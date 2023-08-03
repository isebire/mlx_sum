import argparse
import os
import sys
import torch
from fairseq.models.bart import BARTModel
import calendar
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', help='If set, use the GPU')
    parser.add_argument('--model_path', type=str, default='',
                        help='Model Path')
    parser.add_argument('--srcPath', type=str, default='',
                        help='Source File to Read')
    parser.add_argument('--hypoPath', type=str, default='',
                        help='Hypothesis File to Write')
    parser.add_argument('--minlen', type=int, default=55,
                        help='Min length of summary')
    parser.add_argument('--maxlen', type=int, default=140,
                        help='Max length of summary')
    args = parser.parse_args()

model_path = args.model_path
test_source_path = args.srcPath
test_hypo_path = args.hypoPath

min_len = args.minlen
max_len = args.maxlen

print('Inferring BART for following configuration')
print('srcFile: ', test_source_path, 'model_path: ', model_path, 'hypoFile: ', test_hypo_path, 'min_len: ', min_len,
      'max_len: ', max_len)

# bart = torch.hub.load('pytorch/fairseq', 'bart.large.cnn')
bart = BARTModel.from_pretrained(model_path)
bart.cuda()
bart.eval()
bart.half()
count = 1
bsz = 32

ts = calendar.timegm(time.gmtime())
test_hypo_path = test_hypo_path + str(min_len) + "_" + str(max_len)

# summary length: https://github.com/pytorch/fairseq/issues/1513
# min_len <= output_len <= (max_len_a * input_len) + max_len_b

if os.path.exists(test_hypo_path):
    os.remove(test_hypo_path)

with open(test_source_path) as source, open(test_hypo_path, 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    for sline in source:
        if count % bsz == 0:
            with torch.no_grad():
                hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=max_len, min_len=min_len,
                                               no_repeat_ngram_size=3)

            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
            slines = []

        slines.append(sline.strip())
        count += 1
    if slines != []:
        hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=max_len, min_len=min_len,
                                       no_repeat_ngram_size=3)
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis + '\n')
            fout.flush()

print('Finished inferring BART')
