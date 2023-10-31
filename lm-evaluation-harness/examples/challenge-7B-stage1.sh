#!/bin/bash

# Check arg number
if [[ $# -ne 4 ]]; then
    echo "Four arguments required!" >&2
    exit 2
fi

cd $(dirname "$(dirname "$(realpath $0)")")

# Validate mode
mode=$1
echo 'dev board' | grep -wq $mode
ret=$?
if [[ $ret -ne 0 ]]; then
  echo "Unsupported mode $mode, please specify dev or board!"
  exit $ret
fi

# Prepare paths
model_path=$2
data_dir=$3/${mode}
output_dir=$4/${mode}
mkdir -p ${output_dir}

# Enable offline mode to speed up loading
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Start evaluation
task_fewshot=(
  challenge_mc      25
  challenge_mc_zh   5
  challenge_ma      0
  challenge_summ    0
  challenge_qmsumm  0
)

echo "[MODEL] ${model_path}"
echo "[DATA] ${data_dir}"
echo "[OUT] ${output_dir}"

for ((i=0; i<${#task_fewshot[@]};i+=2)); do
  task=${task_fewshot[$i]}
  fewshot=${task_fewshot[$i+1]}
  echo "[TASK] $task: $fewshot-shot"

  python main.py \
    --model=hf-causal \
    --model_args=pretrained=${model_path},trust_remote_code=True \
    --device=cuda:0 \
    --tasks=${task} \
    --num_fewshot=${fewshot} \
    --batch_size=2 \
    --load_from_disk ${data_dir}/ \
    --output_path=${output_dir}/${task}.json \
    --detail_output_path=${output_dir}/detail/ \
    $(test $mode = board && echo '--infer_only' || echo '')
done

echo "[Done]"