BSc Dissertation (Artificial Intelligence) - Isabel Sebire


The code submission is structured as follows:

* 1_DATASET_AND_EXPLORATORY: Code relating to the exploratory analysis of the Multi-LexSum dataset.
* 2_CLEANING_AND_NER: Code relating to dataset cleaning and NER tagging.
* 3_PRELIM_EXT: Code relating to constructing input representations (includes OREO, and CaseLawBERT classification).
* 4_ABSTRACTIVE: Code relating to finetuning PEGASUS models for abstractive summarisation.
* 5_EVALUATION: Code relating to the evaluation of model outputs.
* 6_QUALITATIVE_EVAL: Utility files for qualitative evaluation.
* CLUSTER_RESOURCES: Utility files for use of external GPU resources.

(Please note, the directory structure and so filepaths in code files may have changed between the time the code was run and this submission.)

_Where the data used came from:_ The primary dataset used is Multi-LexSum - available [here](https://huggingface.co/datasets/allenai/multi_lexsum). Multi-LexSum is distributed under the Open Data Commons Attribution License (ODC-By). The case summaries and metadata are licensed under the Creative Commons Attribution License (CC BY-NC). Although the dataset is not anonymised, the source documents are already publicly accessible and uncopyrighted, as is standard for legal data. This is due to the Courts’ requirement for transparency (the public’s right to know under the US First Amendment) to be balanced with privacy. We note that for the included cases, involved parties can request for cases to proceed by pseudonym in cases with significant privacy interests or involving minors. This information is available in the Multi-LexSum [datasheet](https://arxiv.org/abs/2206.10883), which was thoroughly reviewed.

_How the data was processed:_ The Multi-LexSum dataset was processed as detailed in the thesis - in particular, to construct the inputs to PEGASUS, we cleaned the dataset (2_CLEANING_AND_NER) then constructed the input representations tested (3_PRELIM_EXT). We also constructed entity chains for finetuning certain model configurations (2_CLEANING_AND_NER). 

_Models used_: The pretrained PEGASUS models which we finetuned are freely available on the Huggingface platform. The general-domain PEGASUS model is available [here](https://huggingface.co/google/pegasus-cnn_dailymail) and legal-PEGASUS is available [here](https://huggingface.co/nsi319/legal-pegasus).

_How outputs can be generated:_ Our model outputs (summaries) can be generated using the processed dataset and the pretrained models above, using the code in 4_ABSTRACTIVE. We note that this code was run on an NVIDIA RTX A6000 GPU and thus is unlikely to run without changes on less powerful GPU configurations.

Due to upload size constraints, our finetuned models and processed datasets are not included in this submission - processed datasets and selected trained models are stored in [Huggingface](https://huggingface.co/isebire) and available on request.
All code was written in Python.

