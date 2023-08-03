#!/usr/bin/env bash
# TO BE RUN ON FS

# Check the GPU
nvidia-smi


python -u 4d2_eval_single_run-FS.py \
	--input_dir=/home/fsuser/dissertation/bigeval/inputs \
	--output_dir=/home/fsuser/dissertation/bigeval/outputs \
	| tee 4d2_eval.txt
touch '4d2_done.txt'

python -u 5d2_eval_single_run-FS.py \
	--input_dir=/home/fsuser/dissertation/bigeval/inputs \
	--output_dir=/home/fsuser/dissertation/bigeval/outputs \
	| tee 5d2_eval.txt
touch '5d2_done.txt'
