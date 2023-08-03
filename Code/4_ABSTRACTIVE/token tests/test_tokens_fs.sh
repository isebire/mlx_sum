#!/usr/bin/env bash
# TO BE RUN ON FLUIDSTACK

# Check the GPU
nvidia-smi

echo "Running token test"
python test_tokens.py \
	--input_dir=/home/fsuser/dissertation/inputs \
	--output_dir=/home/fsuser/dissertation/outputs

echo "Running basic token test"
python test_tokens_2.py \
	--input_dir=/home/fsuser/dissertation/inputs \
	--output_dir=/home/fsuser/dissertation/outputs

touch "token_overnight_done.txt"
