#!/bin/bash

export DATA_JUICER_CACHE_HOME=/home/xiejunlin/data1/data_juicer

EXP_NAME=unieval_from_v4_gt_0_7_analysis
CFG_PATH=data-juicer/configs/data_juicer_recipes/dj_comp/visualization.yaml
DATA_PATH=checkpoints/run/run_unieval_dialog_gt_0_7_en_2023-11-23-16-42-45/data/unieval_from_v4_gt_0_7.jsonl
NAME=data_${EXP_NAME}_en
# NAME=run_perplexity_try_en05zh05_2023-10-31-19-09-22
OUTPUT_DIR=data/analysis/${NAME}
OUTPUT_DATA_PATH=${OUTPUT_DIR}/analysis/training_dataset.jsonl

mkdir -p ${OUTPUT_DIR}
cp ./$0 ${OUTPUT_DIR}

# EN_CONFIG_PATH=data-juicer/configs/data_juicer_recipes/exp/${EXP_NAME}.yaml
# EN_CONFIG_PATH=data-juicer/configs/config_all.yaml

echo "[Shell] Running data juicer to process data."
dj-analyze --config ${CFG_PATH} --export_path ${OUTPUT_DIR}/analyser-result.jsonl --dataset_path ${DATA_PATH}
# dj-process --config ${EN_CONFIG_PATH} --export_path ${OUTPUT_DIR}/data/datasets.jsonl --dataset_path data/raw_data/raw_data_en.jsonl