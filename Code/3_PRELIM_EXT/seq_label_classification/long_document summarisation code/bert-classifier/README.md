# Python

* bert_finetuning.py: train, validate and evaluate on test split

* infer_classifier.py: no training, just infer (f.p.) and evaluate on test split

* test_classifier.pt: no training, no evaluation (gold not known). Just infer (f.p.) for unseen data

 
# Shell

## Samples
* run_sample.sh: Runs bert_finetuning.py for style classification data

* run_sts.sh: Runs bert_finetuning.py for sentiment dataset (STS-2)

## train, validate and evaluate
* run_one.sh: Runs bert_finetuning.py for a single configuration (GS)

* run.sh: Runs bert_finetuning.py for a specific configuration (taken from args)

* run_all.sh: Runs run.sh for different combinations of data/type/method

-------
* run-combined.sh: Runs bert_finetuning.py after merging diff datasets for a specific configuration (taken from args)

* run_all-combined.sh: Runs run-combined.sh for different combinations of data/type/method


## infer and evaluate

* infer_one.sh: Runs infer_classifier.py for a single configuration (GS)

* infer.sh: Runs infer_classifier.py for a specific configuration (taken from args)

* infer_all.sh: Runs infer.sh for different combinations of data/type/method

-------
* infer-combined.sh: Runs infer_finetuning.py after merging diff datasets for a specific configuration (taken from args)

* infer_all-combined.sh: Runs infer-combined.sh for different combinations of data/type/method

## predict on unseen data
Change variables in test.sh to change testing data

* test.sh: Runs test_finetuning.py for a specific configuration (taken from args)

* test.sh: Runs test.sh for different combinations of data/type/method

-------
* test-combined.sh: Runs test_classifier.py after merging diff datasets for a specific configuration (taken from args)

* test_all-combined.sh: Runs test-combined.sh for different combinations of data/type/method