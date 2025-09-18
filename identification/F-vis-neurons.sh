#!/bin/bash

RATIO=5
MODEL_NAME="nemo" # llama-3.1
INPUT="activation_mask/${MODEL_NAME}-${RATIO}"
OUTPUT="neurons/${MODEL_NAME}-${RATIO}"

python vis_neurons.py --input_path $INPUT --output_path $OUTPUT