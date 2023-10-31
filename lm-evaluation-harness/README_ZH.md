# 语言模型评估工具


## 安装

```bash
cd lm-evaluation-harness
pip install .
```

## 基本用法

我们在 `examples` 目录中提供了封装的运行脚本，支持一键式评估全部目标任务，请根据赛道选择使用。

例如，执行以下命令在阶段 1 数据上评估 7B 模型：

```bash
bash examples/challenge-7B-stage1.sh \
    <mode> \
    <model_path> \
    <data_path> \
    <output_path>
```

参数：
- \<mode> 可选 `dev` 或 `board`
    - `dev` 模式使用 *dev* 子集数据进行本地评估，仅用于调试
    - `board` 模式使用 *board* 子集数据生成预测结果，用于提交到天池平台
- <model_path> 指定本地模型目录，需为 HuggingFace 格式
- <data_path> 指定评估数据根目录，其下必须包含 *dev* 和 *board* 两个子目录- <output_path> 指定输出根目录

## 本地评测

在向天池提交结果之前，请确保可以成功运行本地评测。

以 [Baichuan2-7B-Base](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base) 模型为例：
- 假设必要的模型文件和权重都已经克隆到本地目录 `./baichaun2-7b-base`
- 下载并解压评测数据到 `./challenge-data`
- 评测结果将会保存到 `./outputs-7b`

执行：

```bash
bash examples/challenge-7B-stage1.sh \
    dev \
    baichaun2-7b-base \
    challenge-data \
    outputs-7b
```

在 `outputs-7b/dev` 文件夹下会产生多个 JSON 文件，包含每个任务的评测结果和关键配置。

同时会伴随产生 `outputs-7b/dev/detail` 目录，详细记录了各任务中每个样本的输出结果。

## 向天池提交

当使用自定义数据微调模型后，请切换到 `board` 模式产生提交文件。

执行：

```bash
bash examples/challenge-7B-stage1.sh \
    board \
    baichaun2-7b-finetuned \
    challenge-data \
    outputs-7b
```

注意，在 `board` 模式下，不会直接产生结果文件，只会产生 `detail` 子目录。

请将 `outputs-7b/board/detail` 目录中的所有中间文件打包为 zip 格式，提交到天池平台。
