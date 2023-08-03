# Python
* bart.py: Infer BART using data and length as arguments

# Shell

## Inference
Uses complete datasets to get summaries

* infer-bart-$dataset$.sh: Runs for particular data (length takes from args)
* infer-all.sh: Runs infer-bart-$dataset$.sh for all datasets and lengths 
---
* infer-ftbart-$dataset$.sh: Runs finetuned checkpoint of BART for particular data (length takes from args)

## Test
Use different subsets of datasets to get summaries
* test-bart-org.sh: Run on original datsets (same as infer-$$ files above) without any extraction
* 