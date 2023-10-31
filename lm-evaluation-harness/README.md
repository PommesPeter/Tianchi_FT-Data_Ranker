# Language Model Evaluation Harness


## Installation

```bash
cd lm-evaluation-harness
pip install .
```

## Basic Usage

We provide several scripts under the `examples` folder for easy evaluation of different tracks.

To evaluate a 7B model on stage-1 data, just run:

```bash
bash examples/challenge-7B-stage1.sh \
    <mode> \
    <model_path> \
    <data_path> \
    <output_path>
```

Arguments:
- \<mode>: `dev` or `board`
    - `dev` mode uses *dev* subset data for local evaluation (debugging purposes)
    - `board` mode uses *board* subset data to generate submission files for Tianchi challenge
- <model_path>: local path of the trained HuggingFace model
- <data_path>: root path of the evaluation data, should containing *dev* and *board* subfolders.
- <output_path>: root path of outputs

## Local Evaluation

Before submitting results to Tianchi, make sure you can run local evaluation successfully.

Take [Baichuan2-7B-Base](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base) model as an example:
- Suppose all model files and weights are cloned to local path `./baichaun2-7b-base`
- Download and extract challenge data to `./challenge-data`
- The evaluation results will be stored in `./outputs-7b`

Execute the following command:

```bash
bash examples/challenge-7B-stage1.sh \
    dev \
    baichaun2-7b-base \
    challenge-data \
    outputs-7b
```

Multiple JSON files will be generated under `outputs-7b/dev`, corresponding to the evaluation results and key configurations of each task.

A `outputs-7b/dev/detail` folder is also accompanied, which records the detailed prediction of each sample in each task.

## Submission

After fine-tuning the model using custom data, switch to `board` mode to generate submission files.

Execute the following command:

```bash
bash examples/challenge-7B-stage1.sh \
    board \
    baichaun2-7b-finetuned \
    challenge-data \
    outputs-7b
```

Note that in `board` mode, the result files will be no longer generated under `outputs-7b/board`.

You just need to package all the intermediate files in the `outputs-7b/board/detail` folder in zip format, and submit it to the Tianchi platform.
