#!/usr/bin/env bash
# TO BE RUN ON FLUIDSTACK

# Check the GPU
nvidia-smi

# Actually do it
echo "Running 2cB"
python 2cB_inference.py \
	--input_dir=/home/fsuser/dissertation/inputs \
	--output_dir=/home/fsuser/dissertation/outputs
touch 'done_2cB.txt'

echo "Running 3cB"
python 3cB_inference.py \
	--input_dir=/home/fsuser/dissertation/inputs \
	--output_dir=/home/fsuser/dissertation/outputs
touch 'done_3cB.txt'

echo "Running 4cB"
python 4cB_inference.py \
	--input_dir=/home/fsuser/dissertation/inputs \
	--output_dir=/home/fsuser/dissertation/outputs
touch 'done_4cB.txt'

echo "Running 5cB"
python 5cB_inference.py \
	--input_dir=/home/fsuser/dissertation/inputs \
	--output_dir=/home/fsuser/dissertation/outputs
touch 'done_5cB.txt'
