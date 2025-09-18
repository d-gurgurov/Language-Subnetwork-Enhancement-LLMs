#!/bin/bash

HF_TOKEN=*

huggingface-cli login --token $HF_TOKEN --add-to-git-credential

export CUDA_VISIBLE_DEVICES=0

rates=(0.01 0.02 0.03 0.04 0.05) #

for i in "${!rates[@]}"; do
    RATE=${rates[$i]}
    RATE_INT=$(printf "%.0f" "$(echo "$RATE * 100" | bc -l)")
    SAVE_PATH="nemo-$RATE_INT"
    echo "Running with top_rate=$RATE, save_path=$SAVE_PATH" # llama_3-1 llama-3.1
    python identify.py --top_rate $RATE --activations "nemo nemo" --save_path "$SAVE_PATH"
done
