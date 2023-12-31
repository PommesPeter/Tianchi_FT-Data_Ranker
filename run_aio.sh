#!/bin/bash

export DATA_JUICER_CACHE_HOME=~/data1/data_juicer
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

NOWTIME=$(date "+%Y-%m-%d-%H-%M-%S")
CUDA_VISIBLE_DEVICES=0,1,2,3
CFG_NAME=all_3sigma_v4_mapper_v3_20231128155600
EXP_NAME=run_dataset_splits
NAME=${EXP_NAME}_en_${NOWTIME}
OUTPUT_DIR=checkpoints/${NAME}
OUTPUT_DATA_PATH=${OUTPUT_DIR}/data/training_dataset.jsonl

mkdir -p ${OUTPUT_DIR}
cp ./$0 ${OUTPUT_DIR}

EN_CONFIG_PATH=data-juicer/configs/data_juicer_recipes/dj_comp/${CFG_NAME}.yaml

# process data
echo "[Shell] Running data juicer to process data."
dj-process --config ${EN_CONFIG_PATH} --export_path ${OUTPUT_DIR}/data/en/datasets_en.jsonl --dataset_path data/raw_data/raw_data_en.jsonl

# sample 3M tokens
echo "[Shell] Running get_train_dataset_1b.py to sample data"
python lm-training/get_train_dataset_1b.py \
    --token_nums 3000000 \
    --ratio 1.0 \
    --en_data_dir ${OUTPUT_DIR}/data/en/datasets_en.jsonl \
    --output_files ${OUTPUT_DATA_PATH}

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

mkdir -p ${output_path}/

WORK_DIR=$(echo `cd $(dirname $0); pwd | xargs dirname`)
# cd ${WORK_DIR}

# Deepspeed
ds_config_file=lm-training/train_scripts/deepspeed_configs/ds_config_stage3.json

# Train Parameter
bs_per_gpu=1
num_nodes=1
nproc_per_node=4
master_port=$(shuf -i 32221-65535 -n 1)

echo "[Shell] Running lm-training"
grad_acc=`expr 256 / ${bs_per_gpu} / ${num_nodes} / ${nproc_per_node}`
# deepspeed --num_gpus ${nproc_per_node} --num_nodes ${num_nodes} --master_port ${master_port} lm-training/train.py \
deepspeed --include localhost:${CUDA_VISIBLE_DEVICES} --master_port ${master_port} lm-training/train.py \
    --model_name_or_path ${model_path} \
    --tokenizer ${tokenizer} \
    --data_path ${data_path} \
    --output_dir ${output_path} \
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
    --deepspeed ${ds_config_file} | tee ${output_path}/training_log.txt

# bash lm-evaluation-harness/examples/challenge-1B-stage1.sh ${NAME}
echo "[Shell] Done"
