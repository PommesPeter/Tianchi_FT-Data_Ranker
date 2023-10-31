import os
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TOKEN_NUMS = 10000000
RATIO = 0.8
EN_DATA_DIR = "raw_data_en.jsonl"
ZH_DATA_DIR = "raw_data_zh.jsonl"
OUTPUT_FILES = "test_7b.jsonl"

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n\n\n### Input:\n\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n\n\n### Response:"
    ),
}

def get_token_count(sample):
    instruction_len = len(enc.tokenize(sample['instruction']))
    input_len = len(enc.tokenize(sample['input']))
    output_len = len(enc.tokenize(sample['output']))
    total_prompt_input_len = instruction_len + input_len + prompt_input_len
    return total_prompt_input_len

def findAllFile(base):
    files = []
    if os.path.isfile(base):
        return [base]
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('jsonl'):
                files.append(os.path.join(root, f))
    return files

enc = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-7B-Base", use_fast=False, trust_remote_code=True)
enc.model_max_length = 1000000000000000019884624838656
prompt_input_len = len(enc.tokenize(PROMPT_DICT["prompt_input"]))

en_files = findAllFile(EN_DATA_DIR)
ds_en = load_dataset('json', data_files=en_files, split='train').shuffle(seed=123)
en_token_nums = TOKEN_NUMS * RATIO if ZH_DATA_DIR else TOKEN_NUMS
zh_token_nums = TOKEN_NUMS - en_token_nums

count = 0
for i in range(len(ds_en)):
    count += get_token_count(ds_en[i])
    print('en num_tokens', i, count)
    if count >= en_token_nums:
        break
ds_en = ds_en.select(range(i+1)).select_columns(['instruction', 'input', 'output'])

if ZH_DATA_DIR:
    zh_files = findAllFile(ZH_DATA_DIR)
    ds_zh = load_dataset('json', data_files=zh_files, split='train').shuffle(seed=123)
    count = 0
    for i in range(len(ds_zh)):
        count += get_token_count(ds_zh[i])
        print('zh num_tokens', i, count)
        if count >= zh_token_nums:
            break
    ds_zh = ds_zh.select(range(i+1)).select_columns(['instruction', 'input', 'output'])
    ds = concatenate_datasets([ds_en, ds_zh]).shuffle(seed=123)
    ds.to_json(OUTPUT_FILES, force_ascii=False)
else:
    ds_en.to_json(OUTPUT_FILES, force_ascii=False)