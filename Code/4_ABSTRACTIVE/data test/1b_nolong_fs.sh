#!/usr/bin/env bash
# TO BE RUN ON FLUIDSTACK

# Check the GPU
nvidia-smi

# Actually do it
echo "Running 1b - nolong"
python 1b_nolong.py \
	--input_dir=/home/fsuser/dissertation/inputs \
	--output_dir=/home/fsuser/dissertation/outputs
