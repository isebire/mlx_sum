# Python

## NGRAM Overlap
* ngram_overlap.py: Visualize overlap
* utils.py: color coding for visualization

## ROUGE
* setting.py: setting file for ROUGE setup (do not change)
* files2rouge.py: base file for ROUGE (do not change)

# Shell

## NGRAM Overlap
* eval_overlap.sh: Runs ngram_overlap.py

## ROUGE

### Full datasets
* cnn-rouge.sh: Calculates rouge after tokenization on CNN data
* rouge.sh: Calculates rouge for GS data (hypothesis, target, out_path taken from args)
* run-rouge.sh: Runs rouge.sh for all datasets and outputs with varying lengths

### Original docs (same as full, no extraction applied)

### Extractive-all (for which target available, directly using topk/agg positive outputs as src). Classifier not inferred)

### Extractive-bart (use classifier's output as src of BART). Only for test split
