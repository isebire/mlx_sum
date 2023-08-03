#!/usr/bin/env bash
# TO BE RUN ON LOCAL

# Actually do it
python -u 1cO_eval_single_run-LOCAL.py \
	--input_dir=/Users/izzy/Desktop/UNI/Diss/Code/5_EVALUATION/pegasus_outputs \
	--output_dir=/Users/izzy/Desktop/UNI/Diss/Code/5_EVALUATION/eval_outputs \
	| tee 1cO_chain.txt
touch '1cO_done.txt'

python -u 2cO_eval_single_run-LOCAL.py \
	--input_dir=/Users/izzy/Desktop/UNI/Diss/Code/5_EVALUATION/pegasus_outputs \
	--output_dir=/Users/izzy/Desktop/UNI/Diss/Code/5_EVALUATION/eval_outputs \
	| tee 2cO_chain.txt
touch '2cO_done.txt'

python -u 3cO_eval_single_run-LOCAL.py \
	--input_dir=/Users/izzy/Desktop/UNI/Diss/Code/5_EVALUATION/pegasus_outputs \
	--output_dir=/Users/izzy/Desktop/UNI/Diss/Code/5_EVALUATION/eval_outputs \
	| tee 3cO_chain.txt
touch '3cO_done.txt'

python -u 4cO_eval_single_run-LOCAL.py \
	--input_dir=/Users/izzy/Desktop/UNI/Diss/Code/5_EVALUATION/pegasus_outputs \
	--output_dir=/Users/izzy/Desktop/UNI/Diss/Code/5_EVALUATION/eval_outputs \
	| tee 4cO_chain.txt
touch '4cO_done.txt'

python -u 5cO_eval_single_run-LOCAL.py \
	--input_dir=/Users/izzy/Desktop/UNI/Diss/Code/5_EVALUATION/pegasus_outputs \
	--output_dir=/Users/izzy/Desktop/UNI/Diss/Code/5_EVALUATION/eval_outputs \
	| tee 5cO_chain.txt
touch '5cO_done.txt'

python -u 1d_eval_single_run-LOCAL.py \
	--input_dir=/Users/izzy/Desktop/UNI/Diss/Code/5_EVALUATION/pegasus_outputs \
	--output_dir=/Users/izzy/Desktop/UNI/Diss/Code/5_EVALUATION/eval_outputs \
	| tee 1d_chain.txt
touch '1d_done.txt'

python -u 2d_eval_single_run-LOCAL.py \
	--input_dir=/Users/izzy/Desktop/UNI/Diss/Code/5_EVALUATION/pegasus_outputs \
	--output_dir=/Users/izzy/Desktop/UNI/Diss/Code/5_EVALUATION/eval_outputs \
	| tee 2d_chain.txt
touch '2d_done.txt'

python -u 3d_eval_single_run-LOCAL.py \
	--input_dir=/Users/izzy/Desktop/UNI/Diss/Code/5_EVALUATION/pegasus_outputs \
	--output_dir=/Users/izzy/Desktop/UNI/Diss/Code/5_EVALUATION/eval_outputs \
	| tee 3d_chain.txt
touch '3d_done.txt'

python -u 4d_eval_single_run-LOCAL.py \
	--input_dir=/Users/izzy/Desktop/UNI/Diss/Code/5_EVALUATION/pegasus_outputs \
	--output_dir=/Users/izzy/Desktop/UNI/Diss/Code/5_EVALUATION/eval_outputs \
	| tee 4d_chain.txt
touch '4d_done.txt'

python -u 5d_eval_single_run-LOCAL.py \
	--input_dir=/Users/izzy/Desktop/UNI/Diss/Code/5_EVALUATION/pegasus_outputs \
	--output_dir=/Users/izzy/Desktop/UNI/Diss/Code/5_EVALUATION/eval_outputs \
	| tee 5d_chain.txt
touch '5d_done.txt'

python -u reprod_eval_single_run-LOCAL.py \
	--input_dir=/Users/izzy/Desktop/UNI/Diss/Code/5_EVALUATION/pegasus_outputs \
	--output_dir=/Users/izzy/Desktop/UNI/Diss/Code/5_EVALUATION/eval_outputs \
	| tee reprod_chain.txt
touch 'reprod_done.txt'

python -u 1cB_eval_single_run-LOCAL.py \
	--input_dir=/Users/izzy/Desktop/UNI/Diss/Code/5_EVALUATION/pegasus_outputs \
	--output_dir=/Users/izzy/Desktop/UNI/Diss/Code/5_EVALUATION/eval_outputs \
	| tee 1cB_chain.txt
touch '1cB_done.txt'

python -u 2cB_eval_single_run-LOCAL.py \
	--input_dir=/Users/izzy/Desktop/UNI/Diss/Code/5_EVALUATION/pegasus_outputs \
	--output_dir=/Users/izzy/Desktop/UNI/Diss/Code/5_EVALUATION/eval_outputs \
	| tee 2cB_chain.txt
touch '2cB_done.txt'

python -u 3cB_eval_single_run-LOCAL.py \
	--input_dir=/Users/izzy/Desktop/UNI/Diss/Code/5_EVALUATION/pegasus_outputs \
	--output_dir=/Users/izzy/Desktop/UNI/Diss/Code/5_EVALUATION/eval_outputs \
	| tee 3cB_chain.txt
touch '3cB_done.txt'

python -u 4cB_eval_single_run-LOCAL.py \
	--input_dir=/Users/izzy/Desktop/UNI/Diss/Code/5_EVALUATION/pegasus_outputs \
	--output_dir=/Users/izzy/Desktop/UNI/Diss/Code/5_EVALUATION/eval_outputs \
	| tee 4cB_chain.txt
touch '4cB_done.txt'

python -u 5cB_eval_single_run-LOCAL.py \
	--input_dir=/Users/izzy/Desktop/UNI/Diss/Code/5_EVALUATION/pegasus_outputs \
	--output_dir=/Users/izzy/Desktop/UNI/Diss/Code/5_EVALUATION/eval_outputs \
	| tee 5cB_chain.txt
touch '5cB_done.txt'

touch 'all_16_done.txt'
