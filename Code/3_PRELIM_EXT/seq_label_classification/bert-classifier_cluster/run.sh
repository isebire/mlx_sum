#!/usr/bin/env bash

# need to have data directories as args as in
# https://github.com/cdt-data-science/cluster-scripts/blob/master/experiments/examples/nlp/src/gpu_predict.py

# ====================
# Activate Anaconda environment
# ====================
source /home/${USER}/miniconda3/bin/activate edidisscluster

# ====================
# RSYNC data from /home/ to /disk/scratch/
# ====================
export SCRATCH_HOME=/disk/scratch/${USER}
export DATA_HOME=${PWD}/inputs_to_send
export DATA_SCRATCH=${SCRATCH_HOME}/pgr/inputs_to_send
mkdir -p ${SCRATCH_HOME}/pgr/inputs_to_send
rsync --archive --update --compress --progress ${DATA_HOME}/ ${DATA_SCRATCH}

echo "Creating directory to save the outputs"
export OUTPUT_DIR=${SCRATCH_HOME}/pgr/outputs
mkdir -p ${OUTPUT_DIR}

# Actually do it
python ft_bert_2.py --gpu --trainFile ${DATA_SCRATCH}/bert_labels_train.tsv --devFile ${DATA_SCRATCH}/bert_labels_validation.tsv --testFile ${DATA_SCRATCH}/bert_labels_test.tsv --outputDir ${OUTPUT_DIR}/test_output --input_dir ${DATA_SCRATCH}

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
