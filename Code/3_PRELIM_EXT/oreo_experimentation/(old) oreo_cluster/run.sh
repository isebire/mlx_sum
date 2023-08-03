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

echo "Creating directories for OREO"
mkdir -p ${OUTPUT_DIR}/json_data
mkdir -p ${OUTPUT_DIR}/bert_data
mkdir -p ${OUTPUT_DIR}/logs

echo "*** running test"
python ${DATA_SCRATCH}/src/labels/test.py

echo "*** running test 2"
split=validation && dataset=mlx && beam=256 && summary_size=6 && src_json_fn=${split}_${dataset}_bert.json && dump_json_fn=mlx_bert-beams_${beam}-steps_${summary_size}.${split}.beam_json && python ${DATA_SCRATCH}/src/labels/test.py

echo "*** Running command 1"

split=validation && dataset=mlx && beam=256 && summary_size=6 && src_json_fn=${split}_${dataset}_bert.json && dump_json_fn=mlx_bert-beams_${beam}-steps_${summary_size}.${split}.beam_json && python ${DATA_SCRATCH}/src/labels/build_beam_json.py --task build_beam_json_for_bert --src ${DATA_SCRATCH}/raw_data/$src_json_fn --save ${OUTPUT_DIR}/json_data/$dump_json_fn --beam $beam --summary_size $summary_size

echo "*** Running command 2"

split=validation && oracle_dist=uniform && beam=256 && summary_size=6 && beam_json_fn=mlx_bert-beams_${beam}-steps_${summary_size}.${split}.beam_json && python ${DATA_SCRATCH}/src/labels/build_bert_json.py --task build_bert_json --src ${OUTPUT_DIR}/json_data/$beam_json_fn --oracle_dist ${oracle_dist} --store_hard_labels --oracle_dist_topk 16

echo "*** Finished running correctly!"

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
