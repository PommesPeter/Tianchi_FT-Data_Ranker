#!/bin/bash

NOWTIME=$(date "+%Y-%m-%d-%H-%M-%S")
# MODEL_NAME=run_keep_long_token_perplexity_refine_v2_html_en_${NOWTIME}
MODEL_NAME=run_keep_long_token_perplexity_refine_v6_1_en_${NOWTIME}

CUDA_DEVICES=0,1,6,7

mkdir -p ../checkpoints/${MODEL_NAME}
cp ./$0 ../checkpoints/${MODEL_NAME}

set -e 
export CUDA_DEVICE_MAX_CONNECTIONS=1

if [ -z $XDG_CACHE_HOME ]; then
    export XDG_CACHE_HOME=$HOME/.cache
fi

if [[ $# -ne 3 ]]; then
    echo "Three arguments required! " >&2
    exit 2
fi

# Model Path
# e.g /home/model/baichuan2-7b/
# model_path=${1} #/path/to/your/model/
model_path="../data/models/falcon-rw-1b" #/path/to/your/model/
tokenizer=${model_path}

# Data Path
# e.g /home/data/train.jsonl
# data_path=${2} # /path/to/your/dataset.jsonl
data_path="../checkpoints/run/run_keep_long_token_perplexity_refine_v6_en_2023-11-06-01-30-32/data/flitered_training_dataset.jsonl" # /path/to/your/dataset.jsonl

# Output Path
# e.g ${WORK_DIR}/checkpoints/baichuan2-7b/
output_path=../checkpoints/${MODEL_NAME} #/path/to/your/output/

mkdir -p ${output_path}/

WORK_DIR=$(echo `cd $(dirname $0); pwd | xargs dirname`)
cd ${WORK_DIR}

# Deepspeed
ds_config_file=${WORK_DIR}/train_scripts/deepspeed_configs/ds_config_stage3.json

# Train Parameter
bs_per_gpu=1
num_nodes=1
nproc_per_node=4
master_port=$(shuf -i 32221-65535 -n 1)

grad_acc=`expr 256 / ${bs_per_gpu} / ${num_nodes} / ${nproc_per_node}`
# deepspeed --num_gpus ${nproc_per_node} --num_nodes ${num_nodes} --master_port ${master_port} train.py \
deepspeed --include localhost:${CUDA_DEVICES} --master_port ${master_port} train.py \
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
