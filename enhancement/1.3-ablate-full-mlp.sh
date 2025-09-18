#!/bin/bash

pip install vllm datasets 
pip install -U accelerate

HF_TOKEN=*
huggingface-cli login --token $HF_TOKEN

pip install numpy==1.26.4 # vllm needs this all of a sudden now !!!

export VLLM_USE_V1=0

MODEL="meta-llama/Llama-3.1-8B"
MODEL_NAME=${MODEL#*/}
RATIO=1
ACTIVATION_MASK="activation_mask/llama-3.1-${RATIO}"
LANGUAGE="mt"

python finetune_full.py --target_lang $LANGUAGE --num_tokens 100000000 --epochs 1 \
                 --output_dir "finetune/${MODEL_NAME}_${RATIO}_${LANGUAGE}_100M_full_mlp" \
                 --model $MODEL --batch_size 2 --fine_tune_mode "mlp_only"