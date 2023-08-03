#!/usr/bin/env bash
# TO BE RUN ON FLUIDSTACK

# Check the GPU
nvidia-smi

# Actually do it
echo "Running 1c - inference on short bert"
python 1c2.py \
	--input_dir=/home/fsuser/dissertation/inputs \
	--output_dir=/home/fsuser/dissertation/outputs
