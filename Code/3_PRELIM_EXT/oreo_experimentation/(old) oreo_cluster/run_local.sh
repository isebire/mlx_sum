#!/usr/bin/env bash

# need to have data directories as args as in
# https://github.com/cdt-data-science/cluster-scripts/blob/master/experiments/examples/nlp/src/gpu_predict.py

# ====================
# Activate Anaconda environment
# ====================

# ====================
# RSYNC data from /home/ to /disk/scratch/
# ====================
export SCRATCH_HOME=.
export OUTPUT_DIR=.


echo "*** Running command 1"

split=validation && dataset=mlx && beam=256 && summary_size=6 && src_json_fn=${split}_${dataset}_bert.json && dump_json_fn=mlx_bert-beams_${beam}-steps_${summary_size}.${split}.beam_json && python src/labels/build_beam_json.py --task build_beam_json_for_bert --src inputs_to_send/raw_data/$src_json_fn --save outputs/json_data/$dump_json_fn --beam $beam --summary_size $summary_size

echo "*** Running command 2"

split=validation && oracle_dist=uniform && beam=256 && summary_size=6 && beam_json_fn=mlx_bert-beams_${beam}-steps_${summary_size}.${split}.beam_json && python src/labels/build_bert_json.py --task build_bert_json --src ${OUTPUT_DIR}/json_data/$beam_json_fn --oracle_dist ${oracle_dist} --store_hard_labels --oracle_dist_topk 16

echo "*** Running command 3"

split=validation && oracle_dist=uniform_top_16_oracle_dist && beam=256 && summary_size=6 && save_dir=mlx_bert-beams_${beam}-steps_${summary_size}-${split}-${oracle_dist}-hard_and_soft && python src/labels/build_bert_json.py --task shard_bert_json --save ${OUTPUT_DIR}/json_data/${save_dir}

echo "*** Running command 4"

split=validation && oracle_dist=uniform_top_16_oracle_dist && beam=256 && summary_size=6 && dir_name=mlx_bert-beams_${beam}-steps_${summary_size}-${split}-${oracle_dist}-hard_and_soft && bert_json_dir=${OUTPUT_DIR}/json_data/${dir_name} && bert_data_dir=${OUTPUT_DIR}/bert_data/${dir_name} && python src/preprocess.py -mode format_to_bert_with_precal_labels -raw_path ${bert_json_dir} -save_path ${bert_data_dir} -lower -n_cpus 1 -log_file ${OUTPUT_DIR}/logs/preprocess.log

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
