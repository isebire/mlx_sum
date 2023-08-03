#!/usr/bin/env bash
# TO BE RUN ON FLUIDSTACK

# Check the GPU
nvidia-smi

python 2d_pegasus.py \
	--input_dir=${DATA_SCRATCH} \
	--output_dir=${OUTPUT_DIR}

touch "2d_run.txt"

python 2cO_pegasus.py \
	--input_dir=${DATA_SCRATCH} \
	--output_dir=${OUTPUT_DIR}

touch "2cO_run.txt"
