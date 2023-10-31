#!/bin/bash

CONFIG_PATH=configs/data_juicer_recipes/alpaca_cot/my-refine-en.yaml

python tools/process_data.py --config ${CONFIG_PATH}