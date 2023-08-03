#!/usr/bin/env bash
# TO BE RUN ON FLUIDSTACK

# Check the GPU
nvidia-smi

# Actually do it
echo "** RUNNING WINDOW VER"

echo "Running 2d"
python 2d_pegasus.py \
	--input_dir=/home/fsuser/dissertation/inputs \
	--output_dir=/home/fsuser/dissertation/outputs

touch 'done_2d.txt'

echo "Running 3d"
python 3d_pegasus.py \
	--input_dir=/home/fsuser/dissertation/inputs \
	--output_dir=/home/fsuser/dissertation/outputs

touch 'done_3d.txt'

echo "Running 4d"
python 4d_pegasus.py \
	--input_dir=/home/fsuser/dissertation/inputs \
	--output_dir=/home/fsuser/dissertation/outputs

touch 'done_4d.txt'

echo "Running 5d"
python 5d_pegasus.py \
	--input_dir=/home/fsuser/dissertation/inputs \
	--output_dir=/home/fsuser/dissertation/outputs

touch 'done_5d.txt'

echo "** RUNNING ORACLE VER"

echo "Running 2cO"
python 2cO_pegasus.py \
	--input_dir=/home/fsuser/dissertation/inputs \
	--output_dir=/home/fsuser/dissertation/outputs

touch 'done_2cO.txt'

echo "Running 3cO"
python 3cO_pegasus.py \
	--input_dir=/home/fsuser/dissertation/inputs \
	--output_dir=/home/fsuser/dissertation/outputs

touch 'done_3cO.txt'

echo "Running 4cO"
python 4cO_pegasus.py \
	--input_dir=/home/fsuser/dissertation/inputs \
	--output_dir=/home/fsuser/dissertation/outputs

touch 'done_4cO.txt'

echo "Running 5cO"
python 5cO_pegasus.py \
	--input_dir=/home/fsuser/dissertation/inputs \
	--output_dir=/home/fsuser/dissertation/outputs

touch 'done_5cO.txt'

touch 'done_all_8.txt'
