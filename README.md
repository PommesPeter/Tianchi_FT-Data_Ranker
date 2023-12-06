# Tianchi_FT-Data_Ranker

本仓库为 [FT-Data Ranker：大语言模型微调数据竞赛 -- 7B模型赛道]([https://tianchi.aliyun.com/competition/entrance/532157](https://tianchi.aliyun.com/competition/entrance/532158)) 比赛的第二名方案。

## 运行步骤

本仓库提供了方便的一键训练脚本 `run_aio.sh`，运行前请执行以下步骤：

1. 在当前项目目录下创建 `checkpoints` 和 `data` 两个文件夹，`data` 文件夹可由官方提供的下载数据脚本 `prepare_data_and_models.sh` 创建。
```bash
mkdir checkpoints
```

2. 修改 `run_aio_7b.sh` 中的 第 12-15 行的变量名，分别表示训练用的设备，配置文件的名字，本次实验的名字。
```shell
#!/bin/bash

...

CUDA_VISIBLE_DEVICES=0,1,2,3 # gpu id
CFG_NAME=<配置文件名字，不需要后缀，默认路径在 `data-juicer/configs/data_juicer_recipes/dj_comp`>
EXP_NAME=<实验的名字，即训练完毕后模型和所使用的数据所在文件夹>

...
```

3. 修改第 76 行的每个节点进程数量，与所运行的 GPU 数量一致。

4. 运行 `run_aio_7b.sh`
```bash
bash run_aio_7b.sh
```

5. 训练完毕后得到的模型和数据文件夹结构如下：
```
checkpoints/run_all_3sigma_v4_en_2023-11-11-17-37-38
├── added_tokens.json
├── config.json
├── configuration_falcon.py
├── data
│   ├── en
│   │   ├── all_3sigma_v4_20231111171400.yaml
│   │   ├── datasets_en.jsonl
│   │   ├── datasets_en_stats.jsonl
│   │   ├── log
│   │   │   └── 20231111173743.txt
│   │   └── trace
│   │       ├── duplicate-document_deduplicator.jsonl
│   │       ├── duplicate-document_simhash_deduplicator.jsonl
│   │       ├── filter-alphanumeric_filter.jsonl
│   │       ├── filter-average_line_length_filter.jsonl
│   │       ├── filter-character_repetition_filter.jsonl
│   │       ├── filter-error_filter.jsonl
│   │       ├── filter-flagged_words_filter.jsonl
│   │       ├── filter-language_id_score_filter.jsonl
│   │       ├── filter-maximum_line_length_filter.jsonl
│   │       ├── filter-output_text_length_filter.jsonl
│   │       ├── filter-perplexity_filter.jsonl
│   │       ├── filter-text_length_filter.jsonl
│   │       ├── filter-text_len_selector.jsonl
│   │       ├── filter-token_num_filter.jsonl
│   │       ├── filter-words_num_filter.jsonl
│   │       ├── mapper-clean_links_mapper.jsonl
│   │       ├── mapper-fix_unicode_mapper.jsonl
│   │       ├── mapper-keyword_mapper.jsonl
│   │       ├── mapper-punctuation_normalization_mapper.jsonl
│   │       └── mapper-whitespace_normalization_mapper.jsonl
│   └── training_dataset.jsonl
├── deepspeed_train_1b.sh
├── generation_config.json
├── merges.txt
├── modeling_falcon.py
├── process_data_only.sh
├── pytorch_model.bin
├── special_tokens_map.json
├── tokenizer_config.json
├── trainer_state.json
├── training_args.bin
├── training_log.txt
└── vocab.json
```

`data` 文件夹包含处理和采样完之后的数据以及 data-juicer 的 trace 文件。`datasets_en.jsonl` 表示经过 data-juicer 处理完之后的数据，`training_dataset.jsonl` 表示采样 3M tokens 之后的数据。其他训练模型产生文件。

6. 测评安装比赛官方提供方式测评即可。

## 数据处理

### Data Juicer 处理

通过分析数据集，发现数据集中存在大量重复且很多格式方面的错误，并且参考 Alpaca-CoT 数据集，我们采用 `data-juicer/configs/data_juicer_recipes/alpaca_cot/alpaca-cot-en-refine.yaml` 作为我们的 baseline，得到初步的结果。

随后继续观察数据集，发现大部分多语种文本都是由 `sharegpt` 这个数据集带来，所以我们认为多语种混合的数据可能会对模型产生影响，我们设计了 `keyword_mapper`，从数据集中筛选出无意义多语种样本的关键词并替换成空格。此外，通过观察数据集还发现存在由于网页爬虫网络错误导致的样本质量过低，故设计 `error_filter` 将含有该关键词的样本过滤掉。同时加入，`clean_links_mapper, fix_unicode_mapper, whitespace_normalization_mapper, punctuation_normalization_mapper` 提高后续计算 language_id_score 的准确性。

再者，通过使用 data-juicer 分析各类指标的分布，以及结合实际样本分析，发现对 `text` 字段的 word_num 在小于 300 的样本质量都很差，估在 baseline 添加 `words_num_filter` 设定 `min_num` 最少要有 300 个单词。此外，我们认为 `output` 字段应该要有至少 10 个文本长度才算有意义，故设计了 `output_text_length_filter`；通过对 perplexity 这个概念的理解，我们认为对于语言模型应该需要困惑度比较低的样本才能有利于模型的学习，故调整了 `perplexity_filter` 的 `max_ppl` 为 1000。对于中文数据，我们设置 `words_num_filter` 设定 `min_num` 最少有 50 个单词。

通过观察训练的代码，我们发现模型在训练之前会对数据进行预处理，会把 token 长度大于 1024 的样本剔除掉，所以考虑这一点我们加入 `token_num_filter` 将样本的 max_num 设置为 1300，进一步筛选有效样本。

最后，通过查阅相关语言模型的训练经验，语言模型应该尽可能需要更多样性的任务类别的数据效果才会更好，也就是数据多样性要丰富，所以我们从 text_len 的角度考虑，增加了 `text_len_selector` 进行采样，保证数据样本在每一个长度区间上都有一定量的数据，保证数据采样的多样性。

### 采样处理

采用中英文各占 50% 的数据来采样。

## 参考资料

- [ALPAGASUS: TRAINING A BETTER ALPACA WITH FEWER DATA](https://arxiv.org/pdf/2307.08701.pdf)
- https://zhuanlan.zhihu.com/p/619241179
- https://zhuanlan.zhihu.com/p/641013454
