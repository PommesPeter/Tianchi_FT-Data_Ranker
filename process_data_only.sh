#!/bin/bash

export PYTHONPATH=/home/xiejunlin/workspace/Tianchi_FT-Data_Ranker/data-juicer
export DATA_JUICER_CACHE_HOME=/home/xiejunlin/data1/data_juicer
export https_proxy=http://uestc.sylin.host:7890
export http_proxy=http://uestc.sylin.host:7890
export all_proxy=socks5://uestc.sylin.host:7890

NOWTIME=$(date "+%Y-%m-%d-%H-%M-%S")
CFG_NAME=all_3sigma_v7_20231117111700
EXP_NAME=all_3sigma_v7_from_v4_data
NAME=run_${EXP_NAME}_en_${NOWTIME}
# NAME=run_all_3sigma_v3_en_2023-11-10-21-04-25
OUTPUT_DIR=checkpoints/run/${NAME}
OUTPUT_DATA_PATH=${OUTPUT_DIR}/data/training_dataset.jsonl

mkdir -p ${OUTPUT_DIR}
cp ./$0 ${OUTPUT_DIR}

EN_CONFIG_PATH=data-juicer/configs/data_juicer_recipes/dj_comp/${CFG_NAME}.yaml
# ZH_CONFIG_PATH=data-juicer/configs/data_juicer_recipes/alpaca_cot/alpaca-cot-zh-refine-perplexity-try.yaml

# process data
echo "[Shell] Running data juicer to process data."
dj-process --config ${EN_CONFIG_PATH} --export_path ${OUTPUT_DIR}/data/en/datasets_en.jsonl --dataset_path checkpoints/run/run_all_3sigma_v4_en_2023-11-11-17-37-38/data/en/datasets_en.jsonl
# dj-process --config ${ZH_CONFIG_PATH} --export_path ${OUTPUT_DIR}/data/zh/datasets_zh.jsonl --dataset_path data/raw_data/raw_data_zh.jsonl

# sample 3M tokens
echo "[Shell] Running get_train_dataset_1b.py to sample data"
python lm-training/get_train_dataset_1b.py \
    --token_nums 3000000 \
    --ratio 1.0 \
    --en_data_dir ${OUTPUT_DIR}/data/en/datasets_en.jsonl \
    --output_files ${OUTPUT_DATA_PATH}
    # --zh_data_dir ${OUTPUT_DIR}/data/zh/datasets_zh.jsonl \
echo "[Shell] Done" 
