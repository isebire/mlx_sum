#!/usr/bin/env bash
# TO BE RUN ON FS

# Check the GPU
nvidia-smi


# Actually do it
python -u 1cO_eval_single_run-FS.py \
	--input_dir=/home/fsuser/dissertation/bigeval/inputs \
	--output_dir=/home/fsuser/dissertation/bigeval/outputs \
	| tee 1c0_eval.txt
touch '1cO_done.txt'

python -u 2cO_eval_single_run-FS.py \
	--input_dir=/home/fsuser/dissertation/bigeval/inputs \
	--output_dir=/home/fsuser/dissertation/bigeval/outputs \
	| tee 2c0_eval.txt
touch '2cO_done.txt'

python -u 3cO_eval_single_run-FS.py \
	--input_dir=/home/fsuser/dissertation/bigeval/inputs \
	--output_dir=/home/fsuser/dissertation/bigeval/outputs \
	| tee 3c0_eval.txt
touch '3cO_done.txt'

python -u 4cO_eval_single_run-FS.py \
	--input_dir=/home/fsuser/dissertation/bigeval/inputs \
	--output_dir=/home/fsuser/dissertation/bigeval/outputs \
	| tee 4c0_eval.txt
touch '4cO_done.txt'

python -u 5cO_eval_single_run-FS.py \
	--input_dir=/home/fsuser/dissertation/bigeval/inputs \
	--output_dir=/home/fsuser/dissertation/bigeval/outputs \
	| tee 5c0_eval.txt
touch '5cO_done.txt'

python -u 1d_eval_single_run-FS.py \
	--input_dir=/home/fsuser/dissertation/bigeval/inputs \
	--output_dir=/home/fsuser/dissertation/bigeval/outputs \
	| tee 1d_eval.txt
touch '1d_done.txt'

python -u 2d_eval_single_run-FS.py \
	--input_dir=/home/fsuser/dissertation/bigeval/inputs \
	--output_dir=/home/fsuser/dissertation/bigeval/outputs \
	| tee 2d_eval.txt
touch '2d_done.txt'

python -u 3d_eval_single_run-FS.py \
	--input_dir=/home/fsuser/dissertation/bigeval/inputs \
	--output_dir=/home/fsuser/dissertation/bigeval/outputs \
	| tee 3d_eval.txt
touch '3d_done.txt'

python -u 4d_eval_single_run-FS.py \
	--input_dir=/home/fsuser/dissertation/bigeval/inputs \
	--output_dir=/home/fsuser/dissertation/bigeval/outputs \
	| tee 4d_eval.txt
touch '4d_done.txt'

python -u 5d_eval_single_run-FS.py \
	--input_dir=/home/fsuser/dissertation/bigeval/inputs \
	--output_dir=/home/fsuser/dissertation/bigeval/outputs \
	| tee 5d_eval.txt
touch '5d_done.txt'

python -u reprod_eval_single_run-FS.py \
	--input_dir=/home/fsuser/dissertation/bigeval/inputs \
	--output_dir=/home/fsuser/dissertation/bigeval/outputs \
	| tee reprod_eval.txt
touch 'reprod_done.txt'

touch 'all_11_done.txt'
