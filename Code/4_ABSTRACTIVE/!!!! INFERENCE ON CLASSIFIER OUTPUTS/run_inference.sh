#!/usr/bin/env bash
# TO BE RUN ON LANDONIA04

# need to have data directories as args as in
# https://github.com/cdt-data-science/cluster-scripts/blob/master/experiments/examples/nlp/src/gpu_predict.py

# Check the GPU
nvidia-smi

# ====================
# Activate Anaconda environment
# ====================
source /home/${USER}/miniconda3/bin/activate pegasus

# ====================
# RSYNC data from /home/ to /disk/scratch/
# ====================
export SCRATCH_HOME=/disk/scratch_big/${USER}
export DATA_HOME=${PWD}/inputs_to_send
export DATA_SCRATCH=${SCRATCH_HOME}/pgr/inputs_to_send
mkdir -p ${SCRATCH_HOME}/pgr/inputs_to_send
rsync --archive --update --compress --progress ${DATA_HOME}/ ${DATA_SCRATCH}

echo "Creating directory to save the outputs"
export OUTPUT_DIR=${SCRATCH_HOME}/pgr/outputs
mkdir -p ${OUTPUT_DIR}

# Actually do it
# *** Change
python pegasus_inference.py \
	--input_dir=${DATA_SCRATCH} \
	--output_dir=${OUTPUT_DIR}

# ====================
# RSYNC data from /disk/scratch/ to /home/. This moves everything we want back onto the distributed file system
# ====================
OUTPUT_HOME=${PWD}/outputs/
mkdir -p ${OUTPUT_HOME}
rsync --archive --update --compress --progress ${OUTPUT_DIR} ${OUTPUT_HOME}

# ====================
# Finally we cleanup after ourselves by deleting what we created on /disk/scratch/
# ====================
rm -rf ${OUTPUT_DIR}
