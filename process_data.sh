#!/bin/bash

export DATA_JUICER_CACHE_HOME=/home/xiejunlin/data1/data_juicer

EXP_NAME=maximum_line
NAME=data_${EXP_NAME}_en
# NAME=run_perplexity_try_en05zh05_2023-10-31-19-09-22
OUTPUT_DIR=data/${NAME}
OUTPUT_DATA_PATH=${OUTPUT_DIR}/analysis/training_dataset.jsonl

mkdir -p ${OUTPUT_DIR}
cp ./$0 ${OUTPUT_DIR}

EN_CONFIG_PATH=data-juicer/configs/data_juicer_recipes/exp/${EXP_NAME}.yaml

echo "[Shell] Running data juicer to process data."
dj-analyze --config ${EN_CONFIG_PATH} --export_path ${OUTPUT_DIR}/analysis/analyser-result.jsonl --dataset_path data/raw_data/raw_data_en.jsonl
dj-process --config ${EN_CONFIG_PATH} --export_path ${OUTPUT_DIR}/data/datasets.jsonl --dataset_path data/raw_data/raw_data_en.jsonl