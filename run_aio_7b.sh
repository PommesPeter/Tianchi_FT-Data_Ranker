#!/bin/bash

export PYTHONPATH=/home/xiejunlin/workspace/Tianchi_FT-Data_Ranker/data-juicer/data_juicer
export DATA_JUICER_CACHE_HOME=/home/xiejunlin/data/data_juicer
export https_proxy=http://uestc.sylin.host:7890
export http_proxy=http://uestc.sylin.host:7890
export all_proxy=socks5://uestc.sylin.host:7890
export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NOWTIME=$(date "+%Y-%m-%d-%H-%M-%S")
CFG_NAME=all_3sigma_v4_20231111171400
EXP_NAME=run_best_v4_for_7b
NAME=${EXP_NAME}_en_${NOWTIME}
# NAME=run_all_sigma_v4_llm_sample_gt_4_2023-11-14-21-34-52
OUTPUT_DIR=checkpoints/run/${NAME}
OUTPUT_DATA_PATH=${OUTPUT_DIR}/data/training_dataset.jsonl

mkdir -p ${OUTPUT_DIR}
cp ./$0 ${OUTPUT_DIR}

EN_CONFIG_PATH=data-juicer/configs/data_juicer_recipes/dj_comp/${CFG_NAME}.yaml
ZH_CONFIG_PATH=data-juicer/configs/data_juicer_recipes/dj_comp/${CFG_NAME}_zh.yaml

# process data
echo "[Shell] Running data juicer to process data."
# dj-process --config ${EN_CONFIG_PATH} --export_path ${OUTPUT_DIR}/data/en/datasets_en.jsonl --dataset_path data/raw_data/raw_data_en.jsonl
# dj-process --config ${ZH_CONFIG_PATH} --export_path ${OUTPUT_DIR}/data/zh/datasets_zh.jsonl --dataset_path data/raw_data/raw_data_zh.jsonl

# sample 3M tokens
echo "[Shell] Running get_train_dataset_7b.py to sample data"
python lm-training/get_train_dataset_7b.py \
    --token_nums 10000000 \
    --ratio 1.0 \
    --en_data_dir ${OUTPUT_DIR}/data/en/datasets_en.jsonl \
    --output_files ${OUTPUT_DATA_PATH}
    # --zh_data_dir ${OUTPUT_DIR}/data/zh/datasets_zh.jsonl \

# training model
# set -e 
export CUDA_DEVICE_MAX_CONNECTIONS=1

if [ -z $XDG_CACHE_HOME ]; then
    export XDG_CACHE_HOME=$HOME/.cache
fi

# Model Path
# e.g /home/model/baichuan2-7b/
# model_path=${1} #/path/to/your/model/
model_path="data/models/falcon-rw-1b" #/path/to/your/model/
tokenizer=${model_path}

# Data Path
# e.g /home/data/train.jsonl
# data_path=${2} # /path/to/your/dataset.jsonl
# data_path="data/test_data/test_en065zh035_1b.jsonl" # /path/to/your/dataset.jsonl
data_path=${OUTPUT_DATA_PATH} # /path/to/your/dataset.jsonl

# Output Path
# e.g ${WORK_DIR}/checkpoints/baichuan2-7b/
output_path=${OUTPUT_DIR} #/path/to/your/output/

lora_path=${output_path}/lora
mkdir -p ${lora_path}/

WORK_DIR=$(echo `cd $(dirname $0); pwd | xargs dirname`)
# cd ${WORK_DIR}

# Deepspeed
ds_config_file=lm-training/train_scripts/deepspeed_configs/ds_config_baichuan2_7b.json

# Train Parameter
bs_per_gpu=1
num_nodes=1
nproc_per_node=8
master_port=$(shuf -i 32221-65535 -n 1)

echo "[Shell] Running lm-training"
grad_acc=`expr 256 / ${bs_per_gpu} / ${num_nodes} / ${nproc_per_node}`
# deepspeed --num_gpus ${nproc_per_node} --num_nodes ${num_nodes} --master_port ${master_port} lm-training/train.py \
deepspeed --include localhost:${CUDA_VISIBLE_DEVICES} --master_port ${master_port} lm-training/train.py \
    --model_name_or_path ${model_path} \
    --tokenizer ${tokenizer} \
    --data_path ${data_path} \
    --output_dir ${lora_path} \
    --per_device_train_batch_size ${bs_per_gpu} \
    --gradient_accumulation_steps ${grad_acc} \
    --lang en \
    --bf16 True \
    --gradient_checkpointing_enable True \
    --num_train_epochs 3 \
    --model_max_length 1024 \
    --learning_rate 2.5e-5 \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps -1 \
    --save_total_limit 999 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --target_modules W_pack \
    --enable_lora True \
    --deepspeed ${ds_config_file} | tee ${output_path}/training_log.txt

# Convert lora to huggingface model
python convert_to_hf.py \
     --model_name_or_path ${model_path} \
     --lora_path ${lora_path}   \
     --output_dir ${output_path} 

echo "[Shell] Done"
