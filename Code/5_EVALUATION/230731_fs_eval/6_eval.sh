#!/usr/bin/env bash
# TO BE RUN ON FS

# Check the GPU
nvidia-smi

# 5 x cB
python -u 1cB_eval_single_run-FS.py \
	--input_dir=/home/fsuser/dissertation/230731_fs_eval/inputs \
	--output_dir=/home/fsuser/dissertation/230731_fs_eval/outputs \
	| tee 1cB_eval.txt
touch '1cB_done.txt'

python -u 2cB_eval_single_run-FS.py \
	--input_dir=/home/fsuser/dissertation/230731_fs_eval/inputs \
	--output_dir=/home/fsuser/dissertation/230731_fs_eval/outputs \
	| tee 2cB_eval.txt
touch '2cB_done.txt'

python -u 3cB_eval_single_run-FS.py \
	--input_dir=/home/fsuser/dissertation/230731_fs_eval/inputs \
	--output_dir=/home/fsuser/dissertation/230731_fs_eval/outputs \
	| tee 3cB_eval.txt
touch '3cB_done.txt'

python -u 4cB_eval_single_run-FS.py \
	--input_dir=/home/fsuser/dissertation/230731_fs_eval/inputs \
	--output_dir=/home/fsuser/dissertation/230731_fs_eval/outputs \
	| tee 4cB_eval.txt
touch '4cB_done.txt'

python -u 5cB_eval_single_run-FS.py \
	--input_dir=/home/fsuser/dissertation/230731_fs_eval/inputs \
	--output_dir=/home/fsuser/dissertation/230731_fs_eval/outputs \
	| tee 5cB_eval.txt
touch '5cB_done.txt'

# 1 x reprod

python -u reprod_eval_single_run-FS.py \
	--input_dir=/home/fsuser/dissertation/230731_fs_eval/inputs \
	--output_dir=/home/fsuser/dissertation/230731_fs_eval/outputs \
	| tee reprod_eval.txt
touch 'reprod_done.txt'

# end

touch 'all_6_done.txt'
