#!/usr/bin/env bash
# TO BE RUN ON FS

# Check the GPU
nvidia-smi


# Actually do it
python 1d_eval_oreo.py \
	--input_dir=/home/fsuser/dissertation/eval/inputs \
	--output_dir=/home/fsuser/dissertation/eval/outputs
