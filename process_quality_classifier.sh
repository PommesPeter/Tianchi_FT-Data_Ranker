#!/bin/bash

export PYTHONPATH=/home/xiejunlin/workspace/Tianchi_FT-Data_Ranker/data-juicer
export DATA_JUICER_CACHE_HOME=/home/xiejunlin/data1/data_juicer
export https_proxy=http://uestc.sylin.host:7890
export http_proxy=http://uestc.sylin.host:7890
export all_proxy=socks5://uestc.sylin.host:7890

NOWTIME=$(date "+%Y-%m-%d-%H-%M-%S")
EXP_NAME=quality_classifier_all_3sigma_v4
NAME=${EXP_NAME}_en_${NOWTIME}
# NAME=run_all_3sigma_v3_en_2023-11-10-21-04-25
DATA_PATH=checkpoints/run/run_all_3sigma_v4_en_2023-11-11-17-37-38/data/en/datasets_en.jsonl
RESULT_DIR=data/${NAME}

mkdir -p ${RESULT_DIR}
cp ./$0 ${RESULT_DIR}

# process data
echo "[Shell] Running data juicer to classify quality."
python data-juicer/tools/quality_classifier/predict.py --dataset_path ${DATA_PATH} --result_path ${RESULT_DIR} --model gpt3 --text_key text --overall_stats true

echo "[Shell] Done" 
